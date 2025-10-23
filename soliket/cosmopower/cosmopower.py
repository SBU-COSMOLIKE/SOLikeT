# VM: code adapted from https://github.com/simonsobs/SOLikeT.git
import os
from typing import Iterable, Tuple
import numpy as np
from cobaya.log import LoggedError
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.theory import Theory
import cosmopower as cp
import math

class CosmoPower(BoltzmannBase):
    def initialize(self) -> None:
        super().initialize()
        if self.network_settings is None:
            raise LoggedError("No network settings were provided.")

        type_ctor = {
            "NN": cp.cosmopower_NN,
            "PCAplusNN": cp.cosmopower_PCAplusNN,
        }
        self.networks = {}
        for spectype, nettype in self.network_settings.items():
            try:
                ctor = type_ctor[nettype["type"]]
            except KeyError as e:
                raise ValueError(f"Unknown network type {nettype['type']} for network {spectype}.") from e

            netpath = os.path.join(self.network_path, nettype["filename"])
            network  = ctor(restore_filename=str(netpath))
            key      = spectype.lower()

            self.networks[key] = {
                "type":           nettype["type"],
                "log":            nettype.get("log", True),
                "network":        network,
                "parameters":     list(network.parameters),
                "lmax":           int(network.modes.max()),
                "has_ell_factor": nettype.get("has_ell_factor", False),
            }

        self.all_parameters = set().union(*(d["parameters"] for d in self.networks.values()))
        self.extra_args.setdefault("lmax", None)
        self.exponent_map = {
            **dict.fromkeys(["tt","te","tb","ee","et","eb","bb","bt","be"], 1.0),
            **dict.fromkeys(["pt","pe","pb","tp","ep","bp"], 1.5),
            "pp": 2.0,
        }

    def calculate(self, state: dict, want_derived: bool = True, **params) -> bool:
        cmb_params = {k: [v] for p, v in params.items() for k in (p, self.translate_param(p))}
        ells = None
        for spectype, net in self.networks.items():
            used = {par: cmb_params.get(par, [params[par]]) for par in net["parameters"]}
            predict = net["network"].ten_to_predictions_np if net["log"] else net["network"].predictions_np
            state[spectype] = predict(used)[0]
            if ells is None:
                ells = net["network"].modes
        state["ell"] = ells.astype(int)
        state["et"] = state["te"]
        return True

    def get_Cl(self, ell_factor: bool = False, units: str = "FIRASmuK2") -> dict:
        cs = self.current_state
        ell = np.arange((self.extra_args.get("lmax") or int(cs["ell"].max()))+1, dtype=int)
        ls  = cs["ell"].astype(int)
        def build(k: str):
            ef   = self.ell_factor(ls, k)
            pref = (1/ef if self.networks[k].get("has_ell_factor") else 1.) * (ef if ell_factor else 1.)
            out = np.zeros_like(ell, dtype=float)
            valid = (ls >= 2) & (ls < ell.size)
            out[ls[valid]] = cs[k][valid] * pref[valid] * self.cmb_unit_factor(k,units,2.7255)
            return out
        return {"ell": ell, **{k: build(k) for k in self.networks}}

    def ell_factor(self, ls: np.ndarray, spectra: str) -> np.ndarray:
        exp  = self.exponent_map.get(spectra)
        return ((ls*(ls+1))**exp/(2*np.pi)**exp) if exp is not None else np.ones_like(ls, float)
   
    def cmb_unit_factor(self, spectra: str,
                        units: str = "1",
                        Tcmb: float = 2.7255) -> float:
        u = self._cmb_unit_factor(units, Tcmb)
        p = 1.0 / math.sqrt(2.0 * math.pi)
        return math.prod(u if c in ("t","e","b") else p if c == "p" else 1.0 for c in spectra.lower())
    
    def get_can_support_parameters(self) -> Iterable[str]:
        return self.all_parameters

    def get_requirements(self) -> Iterable[Tuple[str, str]]:
        rev = {new: old for old, new in self.renames.items()}  # new_name -> old_name
        return [(rev.get(k, k), None) for k in self.all_parameters]

class CosmoPowerDerived(Theory):
    """A theory class that can calculate derived parameters from CosmoPower networks."""

    def initialize(self) -> None:
        super().initialize()

        if self.network_settings is None:
            raise LoggedError("No network settings were provided.")

        netpath = os.path.join(self.network_path, self.network_settings["filename"])

        if self.network_settings["type"] == "NN":
            self.network = cp.cosmopower_NN(restore_filename=netpath)
        elif self.network_settings["type"] == "PCAplusNN":
            self.network = cp.cosmopower_PCAplusNN(
                restore=True, restore_filename=netpath)
        else:
            raise LoggedError(
                f"Unknown network type {self.network_settings['type']}.")

        self.input_parameters = set(self.network.parameters)

        self.log_data = self.network_settings.get("log", False)

        self.log.info(
            f"Loaded CosmoPowerDerived from directory {self.network_path}")
        self.log.info(
            f"CosmoPowerDerived will expect the parameters {self.input_parameters}")
        self.log.info(
            f"CosmoPowerDerived can provide the following parameters: \
                                                            {self.get_can_provide()}.")

    def translate_param(self, p):
        return self.renames.get(p, p)

    def calculate(self, state: dict, want_derived: bool = True, **params) -> bool:
        ## sadly, this syntax not valid until python 3.9
        # input_params = {
        #     p: [params[p]] for p in params
        # } | {
        #     self.translate_param(p): [params[p]] for p in params
        # }
        input_params = {**{
            p: [params[p]] for p in params
        }, **{
            self.translate_param(p): [params[p]] for p in params
        }}

        if self.log_data:
            data = self.network.ten_to_predictions_np(input_params)[0, :]
        else:
            data = self.network.predictions_np(input_params)[0, :]

        for k, v in zip(self.derived_parameters, data):
            if len(k) == 0 or k == "_":
                continue

            state["derived"][k] = v

        return True

    def get_param(self, p) -> float:
        return self.current_state["derived"][self.translate_param(p)]

    def get_can_support_parameters(self) -> Iterable[str]:
        return self.input_parameters

    def get_requirements(self) -> Iterable[Tuple[str, str]]:
        requirements = []
        for k in self.input_parameters:
            if k in self.renames.values():
                for v in self.renames:
                    if self.renames[v] == k:
                        requirements.append((v, None))
                        break
            else:
                requirements.append((k, None))

        return requirements

    def get_can_provide(self) -> Iterable[str]:
        return set([par for par in self.derived_parameters
                    if (len(par) > 0 and not par == "_")])
