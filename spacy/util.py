import functools
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import logging
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

import catalogue
import numpy
import srsly
import thinc
from catalogue import Registry, RegistryError
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from thinc.api import (
    Adam,
    Config,
    ConfigValidationError,
    Model,
    NumpyOps,
    Optimizer,
    get_current_ops,
)

try:
    import cupy.random
except ImportError:
    cupy = None

# These are functions that were previously (v2.x) available from spacy.util
# and have since moved to Thinc. We're importing them here so people's code
# doesn't break, but they should always be imported from Thinc from now on,
# not from spacy.util.
from thinc.api import compounding, decaying, fix_random_seed  # noqa: F401

from . import about
from .compat import CudaStream, cupy, importlib_metadata, is_windows
from .errors import (
    OLD_MODEL_SHORTCUTS,
    ArtifactCommitError,
    ArtifactCommitInterruptedError,
    ArtifactError,  # noqa: F401  (public via spacy.util)
    ArtifactIncompleteError,
    ArtifactIntegrityError,
    ArtifactLockError,
    ArtifactSerializationError,
    ArtifactVersionError,
    Errors,
    Warnings,
)
from .symbols import ORTH

if TYPE_CHECKING:
    # This lets us add type hints for mypy etc. without causing circular imports
    from .language import Language, PipeCallable  # noqa: F401
    from .tokens import Doc, Span  # noqa: F401
    from .vocab import Vocab  # noqa: F401


# fmt: off
OOV_RANK = numpy.iinfo(numpy.uint64).max
DEFAULT_OOV_PROB = -20
LEXEME_NORM_LANGS = ["cs", "da", "de", "el", "en", "grc", "id", "lb", "mk", "pt", "ru", "sr", "ta", "th"]

# Default order of sections in the config file. Not all sections needs to exist,
# and additional sections are added at the end, in alphabetical order.
CONFIG_SECTION_ORDER = ["paths", "variables", "system", "nlp", "components", "corpora", "training", "pretraining", "initialize"]

LANG_ALIASES = {
    "af": ["afr"],
    "am": ["amh"],
    "ar": ["ara"],
    "az": ["aze"],
    "bg": ["bul"],
    "bn": ["ben"],
    "bo": ["bod", "tib"],
    "ca": ["cat"],
    "cs": ["ces", "cze"],
    "da": ["dan"],
    "de": ["deu", "ger"],
    "el": ["ell", "gre"],
    "en": ["eng"],
    "es": ["spa"],
    "et": ["est"],
    "eu": ["eus", "baq"],
    "fa": ["fas", "per"],
    "fi": ["fin"],
    "fo": ["fao"],
    "fr": ["fra", "fre"],
    "ga": ["gle"],
    "gd": ["gla"],
    "gu": ["guj"],
    "he": ["heb", "iw"], # "iw" is the obsolete ISO 639-1 code for Hebrew
    "hi": ["hin"],
    "hr": ["hrv", "scr"], # "scr" is the deprecated ISO 639-2/B for Croatian
    "hu": ["hun"],
    "hy": ["hye"],
    "id": ["ind", "in"], # "in" is the obsolete ISO 639-1 code for Hebrew
    "is": ["isl", "ice"],
    "it": ["ita"],
    "ja": ["jpn"],
    "kn": ["kan"],
    "ko": ["kor"],
    "ky": ["kir"],
    "la": ["lat"],
    "lb": ["ltz"],
    "lg": ["lug"],
    "lt": ["lit"],
    "lv": ["lav"],
    "mk": ["mkd", "mac"],
    "ml": ["mal"],
    "mr": ["mar"],
    "ms": ["msa", "may"],
    "nb": ["nob"],
    "ne": ["nep"],
    "nl": ["nld", "dut"],
    "nn": ["nno"],
    "pl": ["pol"],
    "pt": ["por"],
    "ro": ["ron", "rom", "mo", "mol"], # "mo" and "mol" are deprecated codes for Moldavian
    "ru": ["rus"],
    "sa": ["san"],
    "si": ["sin"],
    "sk": ["slk", "slo"],
    "sl": ["slv"],
    "sq": ["sqi", "alb"],
    "sr": ["srp", "scc"], # "scc" is the deprecated ISO 639-2/B code for Serbian
    "sv": ["swe"],
    "ta": ["tam"],
    "te": ["tel"],
    "th": ["tha"],
    "ti": ["tir"],
    "tl": ["tgl"],
    "tn": ["tsn"],
    "tr": ["tur"],
    "tt": ["tat"],
    "uk": ["ukr"],
    "ur": ["urd"],
    "vi": ["viw"],
    "yo": ["yor"],
    "zh": ["zho", "chi"],

    "xx": ["mul"],
}
# fmt: on

logger = logging.getLogger("spacy")
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setFormatter(
    logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
)
logger.addHandler(logger_stream_handler)


class ENV_VARS:
    CONFIG_OVERRIDES = "SPACY_CONFIG_OVERRIDES"


class registry(thinc.registry):
    languages = catalogue.create("spacy", "languages", entry_points=True)
    architectures = catalogue.create("spacy", "architectures", entry_points=True)
    tokenizers = catalogue.create("spacy", "tokenizers", entry_points=True)
    lemmatizers = catalogue.create("spacy", "lemmatizers", entry_points=True)
    lookups = catalogue.create("spacy", "lookups", entry_points=True)
    displacy_colors = catalogue.create("spacy", "displacy_colors", entry_points=True)
    misc = catalogue.create("spacy", "misc", entry_points=True)
    # Callback functions used to manipulate nlp object etc.
    callbacks = catalogue.create("spacy", "callbacks", entry_points=True)
    batchers = catalogue.create("spacy", "batchers", entry_points=True)
    readers = catalogue.create("spacy", "readers", entry_points=True)
    augmenters = catalogue.create("spacy", "augmenters", entry_points=True)
    loggers = catalogue.create("spacy", "loggers", entry_points=True)
    scorers = catalogue.create("spacy", "scorers", entry_points=True)
    vectors = catalogue.create("spacy", "vectors", entry_points=True)
    # These are factories registered via third-party packages and the
    # spacy_factories entry point. This registry only exists so we can easily
    # load them via the entry points. The "true" factories are added via the
    # Language.factory decorator (in the spaCy code base and user code) and those
    # are the factories used to initialize components via registry.resolve.
    _entry_point_factories = catalogue.create("spacy", "factories", entry_points=True)
    factories = catalogue.create("spacy", "internal_factories")
    # This is mostly used to get a list of all installed models in the current
    # environment. spaCy models packaged with `spacy package` will "advertise"
    # themselves via entry points.
    models = catalogue.create("spacy", "models", entry_points=True)
    cli = catalogue.create("spacy", "cli", entry_points=True)

    @classmethod
    def ensure_populated(cls) -> None:
        """Ensure the registry is populated with all necessary components."""
        from .registrations import REGISTRY_POPULATED, populate_registry

        if not REGISTRY_POPULATED:
            populate_registry()

    @classmethod
    def get_registry_names(cls) -> List[str]:
        """List all available registries."""
        cls.ensure_populated()
        names = []
        for name, value in inspect.getmembers(cls):
            if not name.startswith("_") and isinstance(value, Registry):
                names.append(name)
        return sorted(names)

    @classmethod
    def get(cls, registry_name: str, func_name: str) -> Callable:
        """Get a registered function from the registry."""
        cls.ensure_populated()
        # We're overwriting this classmethod so we're able to provide more
        # specific error messages and implement a fallback to spacy-legacy.
        if not hasattr(cls, registry_name):
            names = ", ".join(cls.get_registry_names()) or "none"
            raise RegistryError(Errors.E892.format(name=registry_name, available=names))
        reg = getattr(cls, registry_name)
        try:
            func = reg.get(func_name)
        except RegistryError:
            if func_name.startswith("spacy."):
                legacy_name = func_name.replace("spacy.", "spacy-legacy.")
                try:
                    return reg.get(legacy_name)
                except catalogue.RegistryError:
                    pass
            available = ", ".join(sorted(reg.get_all().keys())) or "none"
            raise RegistryError(
                Errors.E893.format(
                    name=func_name, reg_name=registry_name, available=available
                )
            ) from None
        return func

    @classmethod
    def find(
        cls, registry_name: str, func_name: str
    ) -> Dict[str, Optional[Union[str, int]]]:
        """Find information about a registered function, including the
        module and path to the file it's defined in, the line number and the
        docstring, if available.

        registry_name (str): Name of the catalogue registry.
        func_name (str): Name of the registered function.
        RETURNS (Dict[str, Optional[Union[str, int]]]): The function info.
        """
        cls.ensure_populated()
        # We're overwriting this classmethod so we're able to provide more
        # specific error messages and implement a fallback to spacy-legacy.
        if not hasattr(cls, registry_name):
            names = ", ".join(cls.get_registry_names()) or "none"
            raise RegistryError(Errors.E892.format(name=registry_name, available=names))
        reg = getattr(cls, registry_name)
        try:
            func_info = reg.find(func_name)
        except RegistryError:
            if func_name.startswith("spacy."):
                legacy_name = func_name.replace("spacy.", "spacy-legacy.")
                try:
                    return reg.find(legacy_name)
                except catalogue.RegistryError:
                    pass
            available = ", ".join(sorted(reg.get_all().keys())) or "none"
            raise RegistryError(
                Errors.E893.format(
                    name=func_name, reg_name=registry_name, available=available
                )
            ) from None
        return func_info

    @classmethod
    def has(cls, registry_name: str, func_name: str) -> bool:
        """Check whether a function is available in a registry."""
        cls.ensure_populated()
        if not hasattr(cls, registry_name):
            return False
        reg = getattr(cls, registry_name)
        if func_name.startswith("spacy."):
            legacy_name = func_name.replace("spacy.", "spacy-legacy.")
            return func_name in reg or legacy_name in reg
        return func_name in reg


class SimpleFrozenDict(dict):
    """Simplified implementation of a frozen dict, mainly used as default
    function or method argument (for arguments that should default to empty
    dictionary). Will raise an error if user or spaCy attempts to add to dict.
    """

    def __init__(self, *args, error: str = Errors.E095, **kwargs) -> None:
        """Initialize the frozen dict. Can be initialized with pre-defined
        values.

        error (str): The error message when user tries to assign to dict.
        """
        super().__init__(*args, **kwargs)
        self.error = error

    def __setitem__(self, key, value):
        raise NotImplementedError(self.error)

    def pop(self, key, default=None):
        raise NotImplementedError(self.error)

    def update(self, other):
        raise NotImplementedError(self.error)


class SimpleFrozenList(list):
    """Wrapper class around a list that lets us raise custom errors if certain
    attributes/methods are accessed. Mostly used for properties like
    Language.pipeline that return an immutable list (and that we don't want to
    convert to a tuple to not break too much backwards compatibility). If a user
    accidentally calls nlp.pipeline.append(), we can raise a more helpful error.
    """

    def __init__(self, *args, error: str = Errors.E927) -> None:
        """Initialize the frozen list.

        error (str): The error message when user tries to mutate the list.
        """
        self.error = error
        super().__init__(*args)

    def append(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def clear(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def extend(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def insert(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def pop(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def remove(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def reverse(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def sort(self, *args, **kwargs):
        raise NotImplementedError(self.error)


def lang_class_is_loaded(lang: str) -> bool:
    """Check whether a Language class is already loaded. Language classes are
    loaded lazily, to avoid expensive setup code associated with the language
    data.

    lang (str): Two-letter language code, e.g. 'en'.
    RETURNS (bool): Whether a Language class has been loaded.
    """
    return lang in registry.languages


def find_matching_language(lang: str) -> Optional[str]:
    """
    Given a two-letter ISO 639-1 or three-letter ISO 639-3 language code,
    find a supported spaCy language.

    Returns the language code if a matching language is available, or None
    if there is no matching language.

    >>> find_matching_language('fra')  # ISO 639-3 code for French
    'fr'
    >>> find_matching_language('fre')  # ISO 639-2/B code for French
    'fr'
    >>> find_matching_language('iw')  # Obsolete ISO 639-1 code for Hebrew
    'he'
    >>> find_matching_language('mo')  # Deprecated code for Moldavian
    'ro'
    >>> find_matching_language('scc')  # Deprecated ISO 639-2/B code for Serbian
    'sr'
    >>> find_matching_language('zxx')
    None
    """
    import spacy.lang  # noqa: F401

    # Check aliases
    for lang_code, aliases in LANG_ALIASES.items():
        if lang in aliases:
            return lang_code

    return None


def get_lang_class(lang: str) -> Type["Language"]:
    """Import and load a Language class.

    lang (str): Two-letter ISO 639-1 or three-letter ISO 639-3 language code, such as 'en' and 'eng'.
    RETURNS (Language): Language class.
    """
    # Check if language is registered / entry point is available
    if lang in registry.languages:
        return registry.languages.get(lang)
    else:
        # Find the language in the spacy.lang subpackage
        try:
            module = importlib.import_module(f".lang.{lang}", "spacy")
        except ImportError as err:
            # Find a matching language. For example, if the language 'eng' is
            # requested, we can use language-matching to load `spacy.lang.en`.
            match = find_matching_language(lang)

            if match:
                lang = match
                module = importlib.import_module(f".lang.{lang}", "spacy")
            else:
                raise ImportError(Errors.E048.format(lang=lang, err=err)) from err
        set_lang_class(lang, getattr(module, module.__all__[0]))  # type: ignore[attr-defined]
    return registry.languages.get(lang)


def set_lang_class(name: str, cls: Type["Language"]) -> None:
    """Set a custom Language class name that can be loaded via get_lang_class.

    name (str): Name of Language class.
    cls (Language): Language class.
    """
    registry.languages.register(name, func=cls)


def ensure_path(path: Any) -> Any:
    """Ensure string is converted to a Path.

    path (Any): Anything. If string, it's converted to Path.
    RETURNS: Path or original argument.
    """
    if isinstance(path, str):
        return Path(path)
    else:
        return path


def load_language_data(path: Union[str, Path]) -> Union[dict, list]:
    """Load JSON language data using the given path as a base. If the provided
    path isn't present, will attempt to load a gzipped version before giving up.

    path (str / Path): The data to load.
    RETURNS: The loaded data.
    """
    path = ensure_path(path)
    if path.exists():
        return srsly.read_json(path)
    path = path.with_suffix(path.suffix + ".gz")
    if path.exists():
        return srsly.read_gzip_json(path)
    raise ValueError(Errors.E160.format(path=path))


def get_module_path(module: ModuleType) -> Path:
    """Get the path of a Python module.

    module (ModuleType): The Python module.
    RETURNS (Path): The path.
    """
    if not hasattr(module, "__module__"):
        raise ValueError(Errors.E169.format(module=repr(module)))
    file_path = Path(cast(os.PathLike, sys.modules[module.__module__].__file__))
    return file_path.parent


# Default value for passed enable/disable values.
_DEFAULT_EMPTY_PIPES = SimpleFrozenList()


def load_model(
    name: Union[str, Path],
    *,
    vocab: Union["Vocab", bool] = True,
    disable: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    enable: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    exclude: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    config: Union[Dict[str, Any], Config] = SimpleFrozenDict(),
) -> "Language":
    """Load a model from a package or data path.

    name (str): Package name or model path.
    vocab (Vocab / True): Optional vocab to pass in on initialization. If True,
        a new Vocab object will be created.
    disable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to disable.
    enable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to enable. All others will be disabled.
    exclude (Union[str, Iterable[str]]):  Name(s) of pipeline component(s) to exclude.
    config (Dict[str, Any] / Config): Config overrides as nested dict or dict
        keyed by section values in dot notation.
    RETURNS (Language): The loaded nlp object.
    """
    kwargs = {
        "vocab": vocab,
        "disable": disable,
        "enable": enable,
        "exclude": exclude,
        "config": config,
    }
    if isinstance(name, str):  # name or string path
        if name.startswith("blank:"):  # shortcut for blank model
            return get_lang_class(name.replace("blank:", ""))()
        if is_package(name):  # installed as package
            return load_model_from_package(name, **kwargs)  # type: ignore[arg-type]
        if Path(name).exists():  # path to model data directory
            return load_model_from_path(Path(name), **kwargs)  # type: ignore[arg-type]
    elif hasattr(name, "exists"):  # Path or Path-like to model data
        return load_model_from_path(name, **kwargs)  # type: ignore[arg-type]
    if name in OLD_MODEL_SHORTCUTS:
        raise IOError(Errors.E941.format(name=name, full=OLD_MODEL_SHORTCUTS[name]))  # type: ignore[index]
    raise IOError(Errors.E050.format(name=name))


def load_model_from_package(
    name: str,
    *,
    vocab: Union["Vocab", bool] = True,
    disable: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    enable: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    exclude: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    config: Union[Dict[str, Any], Config] = SimpleFrozenDict(),
) -> "Language":
    """Load a model from an installed package.

    name (str): The package name.
    vocab (Vocab / True): Optional vocab to pass in on initialization. If True,
        a new Vocab object will be created.
    disable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to disable. Disabled
        pipes will be loaded but they won't be run unless you explicitly
        enable them by calling nlp.enable_pipe.
    enable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to enable. All other
        pipes will be disabled (and can be enabled using `nlp.enable_pipe`).
    exclude (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to exclude. Excluded
        components won't be loaded.
    config (Dict[str, Any] / Config): Config overrides as nested dict or dict
        keyed by section values in dot notation.
    RETURNS (Language): The loaded nlp object.
    """
    cls = importlib.import_module(name)
    return cls.load(
        vocab=vocab, disable=disable, enable=enable, exclude=exclude, config=config
    )  # type: ignore[attr-defined]


def load_model_from_path(
    model_path: Path,
    *,
    meta: Optional[Dict[str, Any]] = None,
    vocab: Union["Vocab", bool] = True,
    disable: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    enable: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    exclude: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    config: Union[Dict[str, Any], Config] = SimpleFrozenDict(),
) -> "Language":
    """Load a model from a data directory path. Creates Language class with
    pipeline from config.cfg and then calls from_disk() with path.

    model_path (Path): Model path.
    meta (Dict[str, Any]): Optional model meta.
    vocab (Vocab / True): Optional vocab to pass in on initialization. If True,
        a new Vocab object will be created.
    disable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to disable. Disabled
        pipes will be loaded but they won't be run unless you explicitly
        enable them by calling nlp.enable_pipe.
    enable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to enable. All other
        pipes will be disabled (and can be enabled using `nlp.enable_pipe`).
    exclude (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to exclude. Excluded
        components won't be loaded.
    config (Dict[str, Any] / Config): Config overrides as nested dict or dict
        keyed by section values in dot notation.
    RETURNS (Language): The loaded nlp object.
    """
    if not model_path.exists():
        raise IOError(Errors.E052.format(path=model_path))
    if not meta:
        meta = get_model_meta(model_path)
    config_path = model_path / "config.cfg"
    overrides = dict_to_dot(config, for_overrides=True)
    config = load_config(config_path, overrides=overrides)
    nlp = load_model_from_config(
        config,
        vocab=vocab,
        disable=disable,
        enable=enable,
        exclude=exclude,
        meta=meta,
    )
    return nlp.from_disk(model_path, exclude=exclude, overrides=overrides)


def load_model_from_config(
    config: Union[Dict[str, Any], Config],
    *,
    meta: Dict[str, Any] = SimpleFrozenDict(),
    vocab: Union["Vocab", bool] = True,
    disable: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    enable: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    exclude: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    auto_fill: bool = False,
    validate: bool = True,
) -> "Language":
    """Create an nlp object from a config. Expects the full config file including
    a section "nlp" containing the settings for the nlp object.

    name (str): Package name or model path.
    meta (Dict[str, Any]): Optional model meta.
    vocab (Vocab / True): Optional vocab to pass in on initialization. If True,
        a new Vocab object will be created.
    disable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to disable. Disabled
        pipes will be loaded but they won't be run unless you explicitly
        enable them by calling nlp.enable_pipe.
    enable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to enable. All other
        pipes will be disabled (and can be enabled using `nlp.enable_pipe`).
    exclude (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to exclude. Excluded
        components won't be loaded.
    auto_fill (bool): Whether to auto-fill config with missing defaults.
    validate (bool): Whether to show config validation errors.
    RETURNS (Language): The loaded nlp object.
    """
    if "nlp" not in config:
        raise ValueError(Errors.E985.format(config=config))
    nlp_config = config["nlp"]
    if "lang" not in nlp_config or nlp_config["lang"] is None:
        raise ValueError(Errors.E993.format(config=nlp_config))
    # This will automatically handle all codes registered via the languages
    # registry, including custom subclasses provided via entry points
    lang_cls = get_lang_class(nlp_config["lang"])
    nlp = lang_cls.from_config(
        config,
        vocab=vocab,
        disable=disable,
        enable=enable,
        exclude=exclude,
        auto_fill=auto_fill,
        validate=validate,
        meta=meta,
    )
    return nlp


def get_sourced_components(
    config: Union[Dict[str, Any], Config],
) -> Dict[str, Dict[str, Any]]:
    """RETURNS (List[str]): All sourced components in the original config,
    e.g. {"source": "en_core_web_sm"}. If the config contains a key
    "factory", we assume it refers to a component factory.
    """
    return {
        name: cfg
        for name, cfg in config.get("components", {}).items()
        if "factory" not in cfg and "source" in cfg
    }


def resolve_dot_names(
    config: Config, dot_names: List[Optional[str]]
) -> Tuple[Any, ...]:
    """Resolve one or more "dot notation" names, e.g. corpora.train.
    The paths could point anywhere into the config, so we don't know which
    top-level section we'll be looking within.

    We resolve the whole top-level section, although we could resolve less --
    we could find the lowest part of the tree.
    """
    # TODO: include schema?
    resolved = {}
    output: List[Any] = []
    errors = []
    for name in dot_names:
        if name is None:
            output.append(name)
        else:
            section = name.split(".")[0]
            # We want to avoid resolving the same thing twice
            if section not in resolved:
                if registry.is_promise(config[section]):
                    # Otherwise we can't resolve [corpus] if it's a promise
                    result = registry.resolve({"config": config[section]})["config"]
                else:
                    result = registry.resolve(config[section])
                resolved[section] = result
            try:
                output.append(dot_to_object(resolved, name))  # type: ignore[arg-type]
            except KeyError:
                msg = f"not a valid section reference: {name}"
                errors.append({"loc": name.split("."), "msg": msg})
    if errors:
        raise ConfigValidationError(config=config, errors=errors)
    return tuple(output)


def load_model_from_init_py(
    init_file: Union[Path, str],
    *,
    vocab: Union["Vocab", bool] = True,
    disable: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    enable: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    exclude: Union[str, Iterable[str]] = _DEFAULT_EMPTY_PIPES,
    config: Union[Dict[str, Any], Config] = SimpleFrozenDict(),
) -> "Language":
    """Helper function to use in the `load()` method of a model package's
    __init__.py.

    vocab (Vocab / True): Optional vocab to pass in on initialization. If True,
        a new Vocab object will be created.
    disable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to disable. Disabled
        pipes will be loaded but they won't be run unless you explicitly
        enable them by calling nlp.enable_pipe.
    enable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to enable. All other
        pipes will be disabled (and can be enabled using `nlp.enable_pipe`).
    exclude (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to exclude. Excluded
        components won't be loaded.
    config (Dict[str, Any] / Config): Config overrides as nested dict or dict
        keyed by section values in dot notation.
    RETURNS (Language): The loaded nlp object.
    """
    model_path = Path(init_file).parent
    meta = get_model_meta(model_path)
    data_dir = f"{meta['lang']}_{meta['name']}-{meta['version']}"
    data_path = model_path / data_dir
    if not model_path.exists():
        raise IOError(Errors.E052.format(path=data_path))
    return load_model_from_path(
        data_path,
        vocab=vocab,
        meta=meta,
        disable=disable,
        enable=enable,
        exclude=exclude,
        config=config,
    )


def load_config(
    path: Union[str, Path],
    overrides: Dict[str, Any] = SimpleFrozenDict(),
    interpolate: bool = False,
) -> Config:
    """Load a config file. Takes care of path validation and section order.

    path (Union[str, Path]): Path to the config file or "-" to read from stdin.
    overrides: (Dict[str, Any]): Config overrides as nested dict or
        dict keyed by section values in dot notation.
    interpolate (bool): Whether to interpolate and resolve variables.
    RETURNS (Config): The loaded config.
    """
    config_path = ensure_path(path)
    config = Config(section_order=CONFIG_SECTION_ORDER)
    if str(config_path) == "-":  # read from standard input
        return config.from_str(
            sys.stdin.read(), overrides=overrides, interpolate=interpolate
        )
    else:
        if not config_path or not config_path.is_file():
            raise IOError(Errors.E053.format(path=config_path, name="config file"))
        return config.from_disk(
            config_path, overrides=overrides, interpolate=interpolate
        )


def load_config_from_str(
    text: str, overrides: Dict[str, Any] = SimpleFrozenDict(), interpolate: bool = False
):
    """Load a full config from a string. Wrapper around Thinc's Config.from_str.

    text (str): The string config to load.
    interpolate (bool): Whether to interpolate and resolve variables.
    RETURNS (Config): The loaded config.
    """
    return Config(section_order=CONFIG_SECTION_ORDER).from_str(
        text, overrides=overrides, interpolate=interpolate
    )


def get_installed_models() -> List[str]:
    """List all model packages currently installed in the environment.

    RETURNS (List[str]): The string names of the models.
    """
    return list(registry.models.get_all().keys())


def get_package_version(name: str) -> Optional[str]:
    """Get the version of an installed package. Typically used to get model
    package versions.

    name (str): The name of the installed Python package.
    RETURNS (str / None): The version or None if package not installed.
    """
    try:
        return importlib_metadata.version(name)  # type: ignore[attr-defined]
    except importlib_metadata.PackageNotFoundError:  # type: ignore[attr-defined]
        return None


def is_compatible_version(
    version: str, constraint: str, prereleases: bool = True
) -> Optional[bool]:
    """Check if a version (e.g. "2.0.0") is compatible given a version
    constraint (e.g. ">=1.9.0,<2.2.1"). If the constraint is a specific version,
    it's interpreted as =={version}.

    version (str): The version to check.
    constraint (str): The constraint string.
    prereleases (bool): Whether to allow prereleases. If set to False,
        prerelease versions will be considered incompatible.
    RETURNS (bool / None): Whether the version is compatible, or None if the
        version or constraint are invalid.
    """
    # Handle cases where exact version is provided as constraint
    if constraint[0].isdigit():
        constraint = f"=={constraint}"
    try:
        spec = SpecifierSet(constraint)
        version = Version(version)  # type: ignore[assignment]
    except (InvalidSpecifier, InvalidVersion):
        return None
    spec.prereleases = prereleases
    return version in spec


def is_unconstrained_version(
    constraint: str, prereleases: bool = True
) -> Optional[bool]:
    # We have an exact version, this is the ultimate constrained version
    if constraint[0].isdigit():
        return False
    try:
        spec = SpecifierSet(constraint)
    except InvalidSpecifier:
        return None
    spec.prereleases = prereleases
    specs = [sp for sp in spec]
    # We only have one version spec and it defines > or >=
    if len(specs) == 1 and specs[0].operator in (">", ">="):
        return True
    # One specifier is exact version
    if any(sp.operator in ("==") for sp in specs):
        return False
    has_upper = any(sp.operator in ("<", "<=") for sp in specs)
    has_lower = any(sp.operator in (">", ">=") for sp in specs)
    # We have a version spec that defines an upper and lower bound
    if has_upper and has_lower:
        return False
    # Everything else, like only an upper version, only a lower version etc.
    return True


def split_requirement(requirement: str) -> Tuple[str, str]:
    """Split a requirement like spacy>=1.2.3 into ("spacy", ">=1.2.3")."""
    req = Requirement(requirement)
    return (req.name, str(req.specifier))


def get_minor_version_range(version: str) -> str:
    """Generate a version range like >=1.2.3,<1.3.0 based on a given version
    (e.g. of spaCy).
    """
    release = Version(version).release
    return f">={version},<{release[0]}.{release[1] + 1}.0"


def get_model_lower_version(constraint: str) -> Optional[str]:
    """From a version range like >=1.2.3,<1.3.0 return the lower pin."""
    try:
        specset = SpecifierSet(constraint)
        for spec in specset:
            if spec.operator in (">=", "==", "~="):
                return spec.version
    except Exception:
        pass
    return None


def is_prerelease_version(version: str) -> bool:
    """Check whether a version is a prerelease version.

    version (str): The version, e.g. "3.0.0.dev1".
    RETURNS (bool): Whether the version is a prerelease version.
    """
    return Version(version).is_prerelease


def get_base_version(version: str) -> str:
    """Generate the base version without any prerelease identifiers.

    version (str): The version, e.g. "3.0.0.dev1".
    RETURNS (str): The base version, e.g. "3.0.0".
    """
    return Version(version).base_version


def get_minor_version(version: str) -> Optional[str]:
    """Get the major + minor version (without patch or prerelease identifiers).

    version (str): The version.
    RETURNS (str): The major + minor version or None if version is invalid.
    """
    try:
        v = Version(version)
    except (TypeError, InvalidVersion):
        return None
    return f"{v.major}.{v.minor}"


def is_minor_version_match(version_a: str, version_b: str) -> bool:
    """Compare two versions and check if they match in major and minor, without
    patch or prerelease identifiers. Used internally for compatibility checks
    that should be insensitive to patch releases.

    version_a (str): The first version
    version_b (str): The second version.
    RETURNS (bool): Whether the versions match.
    """
    a = get_minor_version(version_a)
    b = get_minor_version(version_b)
    return a is not None and b is not None and a == b


def load_meta(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a model meta.json from a path and validate its contents.

    path (Union[str, Path]): Path to meta.json.
    RETURNS (Dict[str, Any]): The loaded meta.
    """
    path = ensure_path(path)
    if not path.parent.exists():
        raise IOError(Errors.E052.format(path=path.parent))
    if not path.exists() or not path.is_file():
        raise IOError(Errors.E053.format(path=path.parent, name="meta.json"))
    meta = srsly.read_json(path)
    for setting in ["lang", "name", "version"]:
        if setting not in meta or not meta[setting]:
            raise ValueError(Errors.E054.format(setting=setting))
    if "spacy_version" in meta:
        if not is_compatible_version(about.__version__, meta["spacy_version"]):
            lower_version = get_model_lower_version(meta["spacy_version"])
            lower_version = get_base_version(lower_version)  # type: ignore[arg-type]
            if lower_version is not None:
                lower_version = "v" + lower_version
            elif "spacy_git_version" in meta:
                lower_version = "git commit " + meta["spacy_git_version"]
            else:
                lower_version = "version unknown"
            warn_msg = Warnings.W095.format(
                model=f"{meta['lang']}_{meta['name']}",
                model_version=meta["version"],
                version=lower_version,
                current=about.__version__,
            )
            warnings.warn(warn_msg)
        if is_unconstrained_version(meta["spacy_version"]):
            warn_msg = Warnings.W094.format(
                model=f"{meta['lang']}_{meta['name']}",
                model_version=meta["version"],
                version=meta["spacy_version"],
                example=get_minor_version_range(about.__version__),
            )
            warnings.warn(warn_msg)
    return meta


def get_model_meta(path: Union[str, Path]) -> Dict[str, Any]:
    """Get model meta.json from a directory path and validate its contents.

    path (str / Path): Path to model directory.
    RETURNS (Dict[str, Any]): The model's meta data.
    """
    model_path = ensure_path(path)
    return load_meta(model_path / "meta.json")


def is_package(name: str) -> bool:
    """Check if string maps to a package installed via pip.

    name (str): Name of package.
    RETURNS (bool): True if installed package, False if not.
    """
    try:
        importlib_metadata.distribution(name)  # type: ignore[attr-defined]
        return True
    except:  # noqa: E722
        return False


def get_package_path(name: str) -> Path:
    """Get the path to an installed package.

    name (str): Package name.
    RETURNS (Path): Path to installed package.
    """
    # Here we're importing the module just to find it. This is worryingly
    # indirect, but it's otherwise very difficult to find the package.
    pkg = importlib.import_module(name)
    return Path(cast(Union[str, os.PathLike], pkg.__file__)).parent


def replace_model_node(model: Model, target: Model, replacement: Model) -> None:
    """Replace a node within a model with a new one, updating refs.

    model (Model): The parent model.
    target (Model): The target node.
    replacement (Model): The node to replace the target with.
    """
    # Place the node into the sublayers
    for node in model.walk():
        if target in node.layers:
            node.layers[node.layers.index(target)] = replacement
    # Now fix any node references
    for node in model.walk():
        for ref_name in node.ref_names:
            if node.maybe_get_ref(ref_name) is target:
                node.set_ref(ref_name, replacement)


def split_command(command: str) -> List[str]:
    """Split a string command using shlex. Handles platform compatibility.
    command (str) : The command to split
    RETURNS (List[str]): The split command.
    """
    return shlex.split(command, posix=not is_windows)


def run_command(
    command: Union[str, List[str]],
    *,
    stdin: Optional[Any] = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command on the command line as a subprocess. If the subprocess
    returns a non-zero exit code, a system exit is performed.
    command (str / List[str]): The command. If provided as a string, the
        string will be split using shlex.split.
    stdin (Optional[Any]): stdin to read from or None.
    capture (bool): Whether to capture the output and errors. If False,
        the stdout and stderr will not be redirected, and if there's an error,
        sys.exit will be called with the return code. You should use capture=False
        when you want to turn over execution to the command, and capture=True
        when you want to run the command more like a function.
    RETURNS (Optional[CompletedProcess]): The process object.
    """
    if isinstance(command, str):
        cmd_list = split_command(command)
        cmd_str = command
    else:
        cmd_list = command
        cmd_str = " ".join(command)
    try:
        ret = subprocess.run(
            cmd_list,
            env=os.environ.copy(),
            input=stdin,
            encoding="utf8",
            check=False,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
        )
    except FileNotFoundError:
        # Indicates the *command* wasn't found, it's an error before the command
        # is run.
        raise FileNotFoundError(
            Errors.E970.format(str_command=cmd_str, tool=cmd_list[0])
        ) from None
    if ret.returncode != 0 and capture:
        message = f"Error running command:\n\n{cmd_str}\n\n"
        message += f"Subprocess exited with status {ret.returncode}"
        if ret.stdout is not None:
            message += f"\n\nProcess log (stdout and stderr):\n\n"
            message += ret.stdout
        error = subprocess.SubprocessError(message)
        error.ret = ret  # type: ignore[attr-defined]
        error.command = cmd_str  # type: ignore[attr-defined]
        raise error
    elif ret.returncode != 0:
        sys.exit(ret.returncode)
    return ret


@contextmanager
def working_dir(path: Union[str, Path]) -> Iterator[Path]:
    """Change current working directory and returns to previous on exit.
    path (str / Path): The directory to navigate to.
    YIELDS (Path): The absolute path to the current working directory. This
        should be used if the block needs to perform actions within the working
        directory, to prevent mismatches with relative paths.
    """
    prev_cwd = Path.cwd()
    current = Path(path).resolve()
    os.chdir(str(current))
    try:
        yield current
    finally:
        os.chdir(str(prev_cwd))


@contextmanager
def make_tempdir() -> Generator[Path, None, None]:
    """Execute a block in a temporary directory and remove the directory and
    its contents at the end of the with block.
    YIELDS (Path): The path of the temp directory.
    """
    d = Path(tempfile.mkdtemp())
    yield d

    # On Windows, git clones use read-only files, which cause permission errors
    # when being deleted. This forcibly fixes permissions.
    def force_remove(rmfunc, path, ex):
        os.chmod(path, stat.S_IWRITE)
        rmfunc(path)

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(str(d), onexc=force_remove)
        else:
            shutil.rmtree(str(d), onerror=force_remove)
    except PermissionError as e:
        warnings.warn(Warnings.W091.format(dir=d, msg=e))


def is_in_jupyter() -> bool:
    """Check if user is running spaCy from a Jupyter or Colab notebook by
    detecting the IPython kernel. Mainly used for the displaCy visualizer.
    RETURNS (bool): True if in Jupyter/Colab, False if not.
    """
    # https://stackoverflow.com/a/39662359/6400719
    # https://stackoverflow.com/questions/15411967
    try:
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":  # type: ignore[name-defined]
            return True  # Jupyter notebook or qtconsole
        if get_ipython().__class__.__module__ == "google.colab._shell":  # type: ignore[name-defined]
            return True  # Colab notebook
    except NameError:
        pass  # Probably standard Python interpreter
    # additional check for Colab
    try:
        import google.colab

        return True  # Colab notebook
    except ImportError:
        pass
    return False


def is_in_interactive() -> bool:
    """Check if user is running spaCy from an interactive Python
    shell. Will return True in Jupyter notebooks too.
    RETURNS (bool): True if in interactive mode, False if not.
    """
    # https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode
    return hasattr(sys, "ps1") or hasattr(sys, "ps2")


def get_object_name(obj: Any) -> str:
    """Get a human-readable name of a Python object, e.g. a pipeline component.

    obj (Any): The Python object, typically a function or class.
    RETURNS (str): A human-readable name.
    """
    if hasattr(obj, "name") and obj.name is not None:
        return obj.name
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "__class__") and hasattr(obj.__class__, "__name__"):
        return obj.__class__.__name__
    return repr(obj)


def is_same_func(func1: Callable, func2: Callable) -> bool:
    """Approximately decide whether two functions are the same, even if their
    identity is different (e.g. after they have been live reloaded). Mostly
    used in the @Language.component and @Language.factory decorators to decide
    whether to raise if a factory already exists. Allows decorator to run
    multiple times with the same function.

    func1 (Callable): The first function.
    func2 (Callable): The second function.
    RETURNS (bool): Whether it's the same function (most likely).
    """
    if not callable(func1) or not callable(func2):
        return False
    if not hasattr(func1, "__qualname__") or not hasattr(func2, "__qualname__"):
        return False
    same_name = func1.__qualname__ == func2.__qualname__
    same_file = inspect.getfile(func1) == inspect.getfile(func2)
    same_code = inspect.getsourcelines(func1) == inspect.getsourcelines(func2)
    return same_name and same_file and same_code


def get_cuda_stream(
    require: bool = False, non_blocking: bool = True
) -> Optional[CudaStream]:
    ops = get_current_ops()
    if CudaStream is None:
        return None
    elif isinstance(ops, NumpyOps):
        return None
    else:
        return CudaStream(non_blocking=non_blocking)


def get_async(stream, numpy_array):
    if cupy is None:
        return numpy_array
    else:
        array = cupy.ndarray(numpy_array.shape, order="C", dtype=numpy_array.dtype)
        array.set(numpy_array, stream=stream)
        return array


def read_regex(path: Union[str, Path]) -> Pattern:
    path = ensure_path(path)
    with path.open(encoding="utf8") as file_:
        entries = file_.read().split("\n")
    expression = "|".join(
        ["^" + re.escape(piece) for piece in entries if piece.strip()]
    )
    return re.compile(expression)


def compile_prefix_regex(entries: Iterable[Union[str, Pattern]]) -> Pattern:
    """Compile a sequence of prefix rules into a regex object.

    entries (Iterable[Union[str, Pattern]]): The prefix rules, e.g.
        spacy.lang.punctuation.TOKENIZER_PREFIXES.
    RETURNS (Pattern): The regex object. to be used for Tokenizer.prefix_search.
    """
    expression = "|".join(["^" + piece for piece in entries if piece.strip()])  # type: ignore[operator, union-attr]
    return re.compile(expression)


def compile_suffix_regex(entries: Iterable[Union[str, Pattern]]) -> Pattern:
    """Compile a sequence of suffix rules into a regex object.

    entries (Iterable[Union[str, Pattern]]): The suffix rules, e.g.
        spacy.lang.punctuation.TOKENIZER_SUFFIXES.
    RETURNS (Pattern): The regex object. to be used for Tokenizer.suffix_search.
    """
    expression = "|".join([piece + "$" for piece in entries if piece.strip()])  # type: ignore[operator, union-attr]
    return re.compile(expression)


def compile_infix_regex(entries: Iterable[Union[str, Pattern]]) -> Pattern:
    """Compile a sequence of infix rules into a regex object.

    entries (Iterable[Union[str, Pattern]]): The infix rules, e.g.
        spacy.lang.punctuation.TOKENIZER_INFIXES.
    RETURNS (regex object): The regex object. to be used for Tokenizer.infix_finditer.
    """
    expression = "|".join([piece for piece in entries if piece.strip()])  # type: ignore[misc, union-attr]
    return re.compile(expression)


def add_lookups(default_func: Callable[[str], Any], *lookups) -> Callable[[str], Any]:
    """Extend an attribute function with special cases. If a word is in the
    lookups, the value is returned. Otherwise the previous function is used.

    default_func (callable): The default function to execute.
    *lookups (dict): Lookup dictionary mapping string to attribute value.
    RETURNS (callable): Lexical attribute getter.
    """
    # This is implemented as functools.partial instead of a closure, to allow
    # pickle to work.
    return functools.partial(_get_attr_unless_lookup, default_func, lookups)


def _get_attr_unless_lookup(
    default_func: Callable[[str], Any], lookups: Dict[str, Any], string: str
) -> Any:
    for lookup in lookups:
        if string in lookup:
            return lookup[string]  # type: ignore[index]
    return default_func(string)


def update_exc(
    base_exceptions: Dict[str, List[dict]], *addition_dicts
) -> Dict[str, List[dict]]:
    """Update and validate tokenizer exceptions. Will overwrite exceptions.

    base_exceptions (Dict[str, List[dict]]): Base exceptions.
    *addition_dicts (Dict[str, List[dict]]): Exceptions to add to the base dict, in order.
    RETURNS (Dict[str, List[dict]]): Combined tokenizer exceptions.
    """
    exc = dict(base_exceptions)
    for additions in addition_dicts:
        for orth, token_attrs in additions.items():
            if not all(isinstance(attr[ORTH], str) for attr in token_attrs):
                raise ValueError(Errors.E055.format(key=orth, orths=token_attrs))
            described_orth = "".join(attr[ORTH] for attr in token_attrs)
            if orth != described_orth:
                raise ValueError(Errors.E056.format(key=orth, orths=described_orth))
        exc.update(additions)
    exc = expand_exc(exc, "'", "’")
    return exc


def expand_exc(
    excs: Dict[str, List[dict]], search: str, replace: str
) -> Dict[str, List[dict]]:
    """Find string in tokenizer exceptions, duplicate entry and replace string.
    For example, to add additional versions with typographic apostrophes.

    excs (Dict[str, List[dict]]): Tokenizer exceptions.
    search (str): String to find and replace.
    replace (str): Replacement.
    RETURNS (Dict[str, List[dict]]): Combined tokenizer exceptions.
    """

    def _fix_token(token, search, replace):
        fixed = dict(token)
        fixed[ORTH] = fixed[ORTH].replace(search, replace)
        return fixed

    new_excs = dict(excs)
    for token_string, tokens in excs.items():
        if search in token_string:
            new_key = token_string.replace(search, replace)
            new_value = [_fix_token(t, search, replace) for t in tokens]
            new_excs[new_key] = new_value
    return new_excs


def normalize_slice(
    length: int, start: int, stop: int, step: Optional[int] = None
) -> Tuple[int, int]:
    if not (step is None or step == 1):
        raise ValueError(Errors.E057)
    if start is None:
        start = 0
    elif start < 0:
        start += length
    start = min(length, max(0, start))
    if stop is None:
        stop = length
    elif stop < 0:
        stop += length
    stop = min(length, max(start, stop))
    return start, stop


def filter_spans(spans: Iterable["Span"]) -> List["Span"]:
    """Filter a sequence of spans and remove duplicates or overlaps. Useful for
    creating named entities (where one token can only be part of one entity) or
    when merging spans with `Retokenizer.merge`. When spans overlap, the (first)
    longest span is preferred over shorter spans.

    spans (Iterable[Span]): The spans to filter.
    RETURNS (List[Span]): The filtered spans.
    """
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens: Set[int] = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def filter_chain_spans(*spans: Iterable["Span"]) -> List["Span"]:
    return filter_spans(itertools.chain(*spans))


def make_first_longest_spans_filter():
    return filter_chain_spans


def to_bytes(getters: Dict[str, Callable[[], bytes]], exclude: Iterable[str]) -> bytes:
    return srsly.msgpack_dumps(to_dict(getters, exclude))


def from_bytes(
    bytes_data: bytes,
    setters: Dict[str, Callable[[bytes], Any]],
    exclude: Iterable[str],
) -> None:
    return from_dict(srsly.msgpack_loads(bytes_data), setters, exclude)  # type: ignore[return-value]


def to_dict(
    getters: Dict[str, Callable[[], Any]], exclude: Iterable[str]
) -> Dict[str, Any]:
    serialized = {}
    for key, getter in getters.items():
        # Split to support file names like meta.json
        if key.split(".")[0] not in exclude:
            serialized[key] = getter()
    return serialized


def from_dict(
    msg: Dict[str, Any],
    setters: Dict[str, Callable[[Any], Any]],
    exclude: Iterable[str],
) -> Dict[str, Any]:
    for key, setter in setters.items():
        # Split to support file names like meta.json
        if key.split(".")[0] not in exclude and key in msg:
            setter(msg[key])
    return msg


# ---------------------------------------------------------------------------
# Transactional artifact storage
#
# Writes never touch the target directory directly. All items are serialized
# into a private staging directory, validated against a manifest and then
# committed as a whole (the manifest is published last). A directory-level
# shared/exclusive lock ensures that cooperating readers either see the
# complete old version or the complete new version, and that concurrent saves
# to the same target are serialized (or rejected with an ArtifactLockError).
#
# On-disk layout (target = e.g. the model directory):
#   <target>/<items...>                 same files/dirs as item-by-item writes
#   <target>/.spacy-manifest.json       commit marker, version and checksums
#   <parent>/.<name>.staging-<txid>/    draft area (removed after commit)
#   <parent>/.<name>.backup-<txid>/     pre-commit backup (removed after commit)
#   <parent>/.<name>.lock               advisory shared/exclusive lock
#
# Directories written before this mechanism have no manifest and continue to
# be read item-by-item exactly as before (legacy mode).
# ---------------------------------------------------------------------------
ARTIFACT_MANIFEST_NAME = ".spacy-manifest.json"
ARTIFACT_MANIFEST_VERSION = 1
ARTIFACT_STAGING_PREFIX_TMPL = ".{name}.staging-"
ARTIFACT_BACKUP_PREFIX_TMPL = ".{name}.backup-"
ARTIFACT_LOCK_NAME_TMPL = ".{name}.lock"
#: Maximum time (seconds) to wait for another process to release the artifact
#: lock before rejecting the operation. Override with SPACY_ARTIFACT_LOCK_TIMEOUT.
ARTIFACT_LOCK_TIMEOUT = 30.0
_LOCK_POLL_INTERVAL = 0.05
_HASH_CHUNK_SIZE = 1024 * 1024

#: Resolved root of the currently active staging area. Nested to_disk calls
#: writing below this root participate in the enclosing transaction instead
#: of starting their own (e.g. Vectors.to_disk writing into the vocab item).
_artifact_write_root: ContextVar[Optional[Path]] = ContextVar(
    "spacy_artifact_write_root", default=None
)
#: Resolved root of a verified, locked read transaction. Nested from_disk
#: calls below this root skip re-verification (the outer reader holds the
#: shared lock and has already checked the manifest).
_artifact_read_root: ContextVar[Optional[Path]] = ContextVar(
    "spacy_artifact_read_root", default=None
)


def _artifact_lock_timeout() -> float:
    try:
        return float(
            os.environ.get("SPACY_ARTIFACT_LOCK_TIMEOUT", ARTIFACT_LOCK_TIMEOUT)
        )
    except (TypeError, ValueError):
        return float(ARTIFACT_LOCK_TIMEOUT)


if os.name == "nt":  # Windows: shared/exclusive locks via LockFileEx
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.LockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_OVERLAPPED),
    ]
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_OVERLAPPED),
    ]
    _kernel32.UnlockFileEx.restype = wintypes.BOOL
    _WIN_LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
    _WIN_LOCKFILE_FAIL_IMMEDIATELY = 0x00000001

    class _PlatformFileLock:
        """A byte-range lock supporting shared and exclusive acquisition."""

        def __init__(self, target: Path, exclusive: bool) -> None:
            self.target = target
            self.lock_path = target.parent / ARTIFACT_LOCK_NAME_TMPL.format(
                name=target.name
            )
            self.exclusive = exclusive
            self.fd: Optional[int] = None
            self._overlapped: Optional["_OVERLAPPED"] = None

        def acquire(self, timeout: float) -> None:
            self.fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o644)
            handle = msvcrt.get_osfhandle(self.fd)
            flags = _WIN_LOCKFILE_FAIL_IMMEDIATELY | (
                _WIN_LOCKFILE_EXCLUSIVE_LOCK if self.exclusive else 0
            )
            deadline = time.monotonic() + timeout
            while True:
                overlapped = _OVERLAPPED()
                if _kernel32.LockFileEx(
                    handle, flags, 0, 1, 0, ctypes.byref(overlapped)
                ):
                    self._overlapped = overlapped
                    return
                if time.monotonic() >= deadline:
                    os.close(self.fd)
                    self.fd = None
                    raise ArtifactLockError(
                        Errors.E1058.format(path=self.target, timeout=timeout),
                        path=self.target,
                        code="E1058",
                    )
                time.sleep(_LOCK_POLL_INTERVAL)

        def release(self) -> None:
            if self.fd is not None:
                handle = msvcrt.get_osfhandle(self.fd)
                if self._overlapped is not None:
                    _kernel32.UnlockFileEx(
                        handle, 0, 1, 0, ctypes.byref(self._overlapped)
                    )
                os.close(self.fd)
                self.fd = None

else:  # POSIX: flock supports shared (LOCK_SH) and exclusive (LOCK_EX) locks
    import fcntl

    class _PlatformFileLock:
        """An advisory flock supporting shared and exclusive acquisition."""

        def __init__(self, target: Path, exclusive: bool) -> None:
            self.target = target
            self.lock_path = target.parent / ARTIFACT_LOCK_NAME_TMPL.format(
                name=target.name
            )
            self.exclusive = exclusive
            self.fd: Optional[int] = None

        def acquire(self, timeout: float) -> None:
            self.fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o644)
            flags = fcntl.LOCK_NB | (fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(self.fd, flags)
                    return
                except OSError:
                    if time.monotonic() >= deadline:
                        os.close(self.fd)
                        self.fd = None
                        raise ArtifactLockError(
                            Errors.E1058.format(path=self.target, timeout=timeout),
                            path=self.target,
                            code="E1058",
                        )
                    time.sleep(_LOCK_POLL_INTERVAL)

        def release(self) -> None:
            if self.fd is not None:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                finally:
                    os.close(self.fd)
                    self.fd = None


@contextmanager
def _artifact_lock(target: Path, exclusive: bool) -> Iterator[None]:
    """Acquire the shared/exclusive lock guarding an artifact directory."""
    lock = _PlatformFileLock(target, exclusive)
    lock.acquire(_artifact_lock_timeout())
    try:
        yield
    finally:
        lock.release()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _artifact_ignored_name(name: str) -> bool:
    """Names excluded from tree checksums (transaction bookkeeping files)."""
    return name == ARTIFACT_MANIFEST_NAME or (
        name.startswith(".")
        and (".staging-" in name or ".backup-" in name or name.endswith(".lock"))
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_:
        for chunk in iter(lambda: file_.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    """Stable, content-defined checksum of a directory tree (git-tree-like)."""
    digest = hashlib.sha256()
    entries = sorted(os.scandir(path), key=lambda entry: entry.name)
    for entry in entries:
        if _artifact_ignored_name(entry.name):
            continue
        is_dir = entry.is_dir(follow_symlinks=False)
        digest.update(b"D " if is_dir else b"F ")
        digest.update(entry.name.encode("utf-8"))
        digest.update(b"\0")
        child = (
            _sha256_tree(Path(entry.path)) if is_dir else _sha256_file(Path(entry.path))
        )
        digest.update(bytes.fromhex(child))
    return digest.hexdigest()


def _item_kind(path: Path) -> Optional[str]:
    if path.is_dir():
        return "dir"
    if path.is_file():
        return "file"
    return None


def _hash_item(path: Path, kind: str) -> str:
    return _sha256_tree(path) if kind == "dir" else _sha256_file(path)


def _build_manifest(root: Path, txid: str) -> Dict[str, Any]:
    """Snapshot the staging directory and hash every entry from the bytes on
    disk (not from memory), so files truncated during write cannot match a
    checksum. Items are recorded by their actual on-disk names: a writer
    receives a path for its logical key but may write elsewhere (e.g. the
    "strings" writer writes "strings.json"), or write nothing at all (e.g.
    empty vectors) - such items are simply absent from the snapshot, and
    readers handle absent items themselves as they always have."""
    items: Dict[str, Any] = {}
    for entry in sorted(os.scandir(root), key=lambda entry: entry.name):
        if _artifact_ignored_name(entry.name):
            continue
        item_path = Path(entry.path)
        kind = "dir" if entry.is_dir(follow_symlinks=False) else "file"
        items[entry.name] = {
            "type": kind,
            "sha256": _hash_item(item_path, kind),
        }
    return {
        "manifest_version": ARTIFACT_MANIFEST_VERSION,
        "spacy_version": about.__version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "txid": txid,
        "items": items,
    }


def _write_manifest(root: Path, manifest: Dict[str, Any]) -> None:
    tmp_path = root / f"{ARTIFACT_MANIFEST_NAME}.tmp-{uuid.uuid4().hex}"
    srsly.write_json(tmp_path, manifest, indent=2)
    os.replace(tmp_path, root / ARTIFACT_MANIFEST_NAME)


def _parse_manifest(target: Path, manifest_path: Path) -> Dict[str, Any]:
    """Read and validate the manifest, raising typed errors for each distinct
    failure mode."""
    try:
        manifest = srsly.json_loads(manifest_path.read_bytes())
    except Exception as err:
        raise ArtifactCommitInterruptedError(
            Errors.E1064.format(path=target, reason=f"{type(err).__name__}: {err}"),
            path=target,
            code="E1064",
        ) from err
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("manifest_version"), int
    ):
        raise ArtifactCommitInterruptedError(
            Errors.E1064.format(
                path=target, reason="the manifest is not a valid manifest object"
            ),
            path=target,
            code="E1064",
        )
    found = manifest["manifest_version"]
    if found > ARTIFACT_MANIFEST_VERSION:
        reason = (
            "The artifact was written with a newer manifest format, so this "
            "reader is stale. Upgrade spaCy to read it."
        )
        kind = "artifact-newer"
    elif found < ARTIFACT_MANIFEST_VERSION:
        reason = (
            "The artifact uses an older, no longer supported manifest format, "
            "so the artifact itself is stale. Re-export it with a current "
            "spaCy version."
        )
        kind = "artifact-older"
    else:
        return manifest
    raise ArtifactVersionError(
        Errors.E1059.format(
            path=target,
            found=found,
            supported=ARTIFACT_MANIFEST_VERSION,
            reason=reason,
        ),
        path=target,
        code="E1059",
        found=found,
        supported=ARTIFACT_MANIFEST_VERSION,
        kind=kind,
    )


def _verify_manifest(target: Path, manifest: Dict[str, Any]) -> None:
    """Check that every recorded item exists with the right type and
    checksum. Missing/wrong-typed items and checksum mismatches get different
    exception types."""
    items = manifest.get("items")
    if not isinstance(items, dict):
        raise ArtifactCommitInterruptedError(
            Errors.E1064.format(
                path=target, reason="the manifest has no 'items' table"
            ),
            path=target,
            code="E1064",
        )
    for key, item_manifest in items.items():
        item_path = target / key
        expected_type = item_manifest.get("type")
        if not item_path.exists() and not item_path.is_symlink():
            raise ArtifactIncompleteError(
                Errors.E1061.format(
                    path=target,
                    item=key,
                    expected_type=expected_type,
                ),
                path=target,
                code="E1061",
                item=key,
                expected_type=expected_type,
                actual_type="missing",
            )
        actual_type = _item_kind(item_path)
        if actual_type != expected_type:
            raise ArtifactIncompleteError(
                Errors.E1061.format(
                    path=target,
                    item=key,
                    expected_type=expected_type,
                ),
                path=target,
                code="E1061",
                item=key,
                expected_type=expected_type,
                actual_type=actual_type,
            )
        actual_hash = _hash_item(item_path, actual_type)
        expected_hash = item_manifest.get("sha256")
        if actual_hash != expected_hash:
            raise ArtifactIntegrityError(
                Errors.E1060.format(
                    path=target,
                    item=key,
                    expected=expected_hash,
                    actual=actual_hash,
                ),
                path=target,
                code="E1060",
                item=key,
                expected=expected_hash,
                actual=actual_hash,
            )


def _artifact_markers(target: Path) -> Tuple[List[Path], List[Path]]:
    """Return (staging dirs, backup dirs) left behind by previous transactions
    for this target."""
    if target.name:
        staging_prefix = ARTIFACT_STAGING_PREFIX_TMPL.format(name=target.name)
        backup_prefix = ARTIFACT_BACKUP_PREFIX_TMPL.format(name=target.name)
    else:
        staging_prefix = backup_prefix = ""
    stagings: List[Path] = []
    backups: List[Path] = []
    if staging_prefix and target.parent.exists():
        for entry in target.parent.iterdir():
            if entry.name.startswith(staging_prefix) and entry.is_dir():
                stagings.append(entry)
            elif entry.name.startswith(backup_prefix) and entry.is_dir():
                backups.append(entry)
    return stagings, backups


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _committed_txid(target: Path) -> Optional[str]:
    manifest_path = target / ARTIFACT_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        manifest = srsly.json_loads(manifest_path.read_bytes())
    except Exception:
        return None
    return manifest.get("txid") if isinstance(manifest, dict) else None


def _recover_artifacts(target: Path) -> None:
    """Resolve leftovers of transactions whose writer died mid-commit.

    A backup whose txid matches the committed manifest is stale garbage (the
    commit completed; only cleanup was missed). Any other backup means the
    commit point was never reached, so the backed-up items are moved back to
    restore the last complete version. Items the dead transaction was adding
    for the first time (and that are therefore not in the backup) are removed
    using the staging manifest. Staging dirs themselves were never visible
    and are deleted afterwards. Must be called under the exclusive lock."""
    stagings, backups = _artifact_markers(target)
    current_txid = _committed_txid(target) if target.exists() else None
    current_manifest: Optional[Dict[str, Any]] = None
    if target.exists() and (target / ARTIFACT_MANIFEST_NAME).exists():
        try:
            current_manifest = srsly.json_loads(
                (target / ARTIFACT_MANIFEST_NAME).read_bytes()
            )
        except Exception:
            current_manifest = None
    current_items = set(
        current_manifest.get("items", {}).keys()
        if isinstance(current_manifest, dict)
        else []
    )
    restored: Set[str] = set()
    for backup in backups:
        txid = backup.name.rsplit("-", 1)[-1]
        if current_txid is not None and txid == current_txid:
            shutil.rmtree(backup, ignore_errors=True)
            continue
        try:
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
            # Restore items first, manifest last so a reader never observes a
            # manifest describing a half-restored tree.
            entries = sorted(
                backup.iterdir(),
                key=lambda entry: entry.name == ARTIFACT_MANIFEST_NAME,
            )
            for entry in entries:
                dest = target / entry.name
                if dest.exists() or dest.is_symlink():
                    _remove_path(dest)
                os.replace(entry, dest)
                restored.add(entry.name)
        except OSError as err:
            raise ArtifactCommitInterruptedError(
                Errors.E1062.format(
                    path=target,
                    detail=f"unresolved backup '{backup.name}'",
                    reason=str(err),
                ),
                path=target,
                code="E1062",
            ) from err
        shutil.rmtree(backup, ignore_errors=True)
    for staging in stagings:
        # Remove half-published items the dead transaction was adding for the
        # first time: they were never committed, so they are neither part of
        # a restored backup nor of the currently committed manifest.
        try:
            staged_manifest = srsly.json_loads(
                (staging / ARTIFACT_MANIFEST_NAME).read_bytes()
            )
            staged_items = staged_manifest.get("items", {})
            if target.exists() and isinstance(staged_items, dict):
                for name in staged_items:
                    dest = target / name
                    if (
                        name not in restored
                        and name not in current_items
                        and (dest.exists() or dest.is_symlink())
                    ):
                        _remove_path(dest)
        except Exception:
            pass
        shutil.rmtree(staging, ignore_errors=True)


def _commit_artifacts(
    target: Path, staging: Path, item_names: List[str], txid: str
) -> None:
    """Publish the validated staging directory as one transaction.

    For a new target the whole directory is renamed into place atomically.
    For an existing target each replaced item is parked in a backup directory
    and the manifest is published last; on any failure everything is moved
    back so the target contains the complete previous version. Items the new
    write does not produce are left untouched (matching the legacy
    item-by-item behavior)."""
    if not target.exists():
        # Same-directory rename: atomic on POSIX and Windows, and the
        # manifest travels with the directory.
        os.replace(staging, target)
        return
    backup = target.parent / (
        ARTIFACT_BACKUP_PREFIX_TMPL.format(name=target.name) + txid
    )
    backup.mkdir()
    names = list(item_names) + [ARTIFACT_MANIFEST_NAME]
    moved_out: List[str] = []
    moved_in: List[str] = []
    try:
        # Phase 1: park the previous versions (old manifest first).
        for name in names:
            old_path = target / name
            if old_path.exists() or old_path.is_symlink():
                os.replace(old_path, backup / name)
                moved_out.append(name)
        # Phase 2: move the new items into place.
        for name in item_names:
            os.replace(staging / name, target / name)
            moved_in.append(name)
        # Phase 3: the commit point - publish the new manifest last.
        os.replace(staging / ARTIFACT_MANIFEST_NAME, target / ARTIFACT_MANIFEST_NAME)
    except OSError as err:
        # Roll back: discard new items, restore old ones, manifest last.
        for name in moved_in:
            new_path = target / name
            if new_path.exists() or new_path.is_symlink():
                _remove_path(new_path)
        for name in moved_out:
            if name == ARTIFACT_MANIFEST_NAME:
                continue
            old_path = backup / name
            if old_path.exists() or old_path.is_symlink():
                dest = target / name
                if dest.exists() or dest.is_symlink():
                    _remove_path(dest)
                os.replace(old_path, dest)
        if ARTIFACT_MANIFEST_NAME in moved_out:
            dest = target / ARTIFACT_MANIFEST_NAME
            if dest.exists():
                _remove_path(dest)
            os.replace(backup / ARTIFACT_MANIFEST_NAME, dest)
        shutil.rmtree(backup, ignore_errors=True)
        raise ArtifactCommitError(
            Errors.E1065.format(path=target, reason=str(err)),
            path=target,
            code="E1065",
        ) from err
    except BaseException:
        # Roll back for non-OS failures too (e.g. KeyboardInterrupt), then
        # preserve the original exception type.
        for name in moved_in:
            new_path = target / name
            if new_path.exists() or new_path.is_symlink():
                _remove_path(new_path)
        for name in moved_out:
            if name == ARTIFACT_MANIFEST_NAME:
                continue
            old_path = backup / name
            if old_path.exists() or old_path.is_symlink():
                dest = target / name
                if dest.exists() or dest.is_symlink():
                    _remove_path(dest)
                os.replace(old_path, dest)
        if ARTIFACT_MANIFEST_NAME in moved_out:
            dest = target / ARTIFACT_MANIFEST_NAME
            if dest.exists():
                _remove_path(dest)
            os.replace(backup / ARTIFACT_MANIFEST_NAME, dest)
        shutil.rmtree(backup, ignore_errors=True)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def to_disk(
    path: Union[str, Path],
    writers: Dict[str, Callable[[Path], None]],
    exclude: Iterable[str],
) -> Path:
    """Serialize artifacts to a directory transactionally.

    All writers run into a private staging directory; the staged items are
    checksummed into a manifest and re-validated, after which the whole
    transaction is committed to the target directory (manifest last). If a
    writer fails, the target is never touched; if the commit fails, the target
    is rolled back to its previous complete state. Concurrent saves to the
    same target are serialized with a directory lock; a save that cannot
    acquire the lock in time fails with an ArtifactLockError without touching
    the target.

    Nested to_disk calls that write below an active staging area participate
    in the enclosing transaction instead of starting a new one.

    path (str / Path): Path to a directory, which will be created if it
        doesn't exist.
    writers (Dict[str, Callable[[Path], None]]): Serializers keyed by item
        name, called with the path to write each item to.
    exclude (Iterable[str]): Names of items to skip.
    RETURNS (Path): The resolved target path.
    """
    path = ensure_path(path).resolve()
    write_root = _artifact_write_root.get()
    if write_root is not None and _is_relative_to(path, write_root):
        # Inside an active transaction: write directly, the outer call
        # validates and commits atomically.
        path.mkdir(parents=True, exist_ok=True)
        for key, writer in writers.items():
            # Split to support file names like meta.json
            if key.split(".")[0] not in exclude:
                writer(path / key)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with _artifact_lock(path, exclusive=True):
        # Finish or undo transactions of writers that died previously.
        _recover_artifacts(path)
        txid = uuid.uuid4().hex
        staging = path.parent / (
            ARTIFACT_STAGING_PREFIX_TMPL.format(name=path.name) + txid
        )
        staging.mkdir()
        staging_root = staging.resolve()
        keys: List[str] = []
        write_token = _artifact_write_root.set(staging_root)
        try:
            for key, writer in writers.items():
                # Split to support file names like meta.json
                if key.split(".")[0] not in exclude:
                    keys.append(key)
                    writer(staging / key)
            # All serialization succeeded: build the manifest from the bytes
            # actually on disk, publish it inside the staging area and verify
            # the staging area against the manifest before committing.
            manifest = _build_manifest(staging, txid)
            _write_manifest(staging, manifest)
            staged_manifest = _parse_manifest(staging, staging / ARTIFACT_MANIFEST_NAME)
            try:
                # Independent re-read of the draft: a failure here is a
                # writer-side failure before the commit point, never an
                # artifact-side integrity problem.
                _verify_manifest(staging, staged_manifest)
            except ArtifactError as err:
                raise ArtifactSerializationError(
                    Errors.E1063.format(
                        path=staging,
                        reason=(
                            f"validation of staged item '{err.details.get('item')}' "
                            f"failed ({type(err).__name__})."
                        ),
                    ),
                    path=staging,
                    code="E1063",
                    details=err.details,
                ) from err
            _commit_artifacts(path, staging, list(manifest["items"].keys()), txid)
        finally:
            _artifact_write_root.reset(write_token)
            # After a successful rename the staging path no longer exists;
            # after a merge commit it is an empty directory. Either way make
            # sure no draft is left behind next to the target.
            shutil.rmtree(staging, ignore_errors=True)
    return path


def _dispatch_readers(
    path: Path,
    readers: Dict[str, Callable[[Path], None]],
    exclude: Iterable[str],
) -> None:
    for key, reader in readers.items():
        # Split to support file names like meta.json
        if key.split(".")[0] not in exclude:
            reader(path / key)


def from_disk(
    path: Union[str, Path],
    readers: Dict[str, Callable[[Path], None]],
    exclude: Iterable[str],
) -> Path:
    """Load artifacts from a directory with integrity verification.

    Directories with a manifest are verified before any reader runs: manifest
    version mismatches raise ArtifactVersionError, missing items raise
    ArtifactIncompleteError and checksum mismatches raise
    ArtifactIntegrityError. These are distinct from reading a legacy
    directory (no manifest), which keeps the previous item-by-item behavior
    unchanged. A shared lock blocks readers while a writer is committing, so
    readers always observe a complete old or complete new version.

    path (str / Path): A path to a directory.
    readers (Dict[str, Callable[[Path], None]]): Deserializers keyed by item
        name, called with the path to read each item from.
    exclude (Iterable[str]): Names of items to skip.
    RETURNS (Path): The resolved target path.
    """
    path = ensure_path(path).resolve()
    read_root = _artifact_read_root.get()
    if read_root is not None and _is_relative_to(path, read_root):
        # Outer reader holds the lock and has verified everything.
        _dispatch_readers(path, readers, exclude)
        return path
    if not path.exists():
        # Nothing committed here: behave exactly like the legacy reader.
        _dispatch_readers(path, readers, exclude)
        return path
    # Leftover backup directories mean a previous commit may have been
    # interrupted: take the exclusive lock briefly and recover (this also
    # waits for a commit that is currently in progress).
    _, backups = _artifact_markers(path)
    exclusive = bool(backups)
    with _artifact_lock(path, exclusive=exclusive):
        if exclusive:
            _recover_artifacts(path)
        manifest_path = path / ARTIFACT_MANIFEST_NAME
        if not manifest_path.exists():
            # Legacy artifact (written without transactions): read exactly
            # as before, without verification.
            _dispatch_readers(path, readers, exclude)
            return path
        manifest = _parse_manifest(path, manifest_path)
        _verify_manifest(path, manifest)
        token = _artifact_read_root.set(path)
        try:
            _dispatch_readers(path, readers, exclude)
        finally:
            _artifact_read_root.reset(token)
    return path


def import_file(name: str, loc: Union[str, Path]) -> ModuleType:
    """Import module from a file. Used to load models from a directory.

    name (str): Name of module to load.
    loc (str / Path): Path to the file.
    RETURNS: The loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, str(loc))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def minify_html(html: str) -> str:
    """Perform a template-specific, rudimentary HTML minification for displaCy.
    Disclaimer: NOT a general-purpose solution, only removes indentation and
    newlines.

    html (str): Markup to minify.
    RETURNS (str): "Minified" HTML.
    """
    return html.strip().replace("    ", "").replace("\n", "")


def escape_html(text: str) -> str:
    """Replace <, >, &, " with their HTML encoded representation. Intended to
    prevent HTML errors in rendered displaCy markup.

    text (str): The original text.
    RETURNS (str): Equivalent text to be safely used within HTML.
    """
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    return text


def get_words_and_spaces(
    words: Iterable[str], text: str
) -> Tuple[List[str], List[bool]]:
    """Given a list of words and a text, reconstruct the original tokens and
    return a list of words and spaces that can be used to create a Doc. This
    can help recover destructive tokenization that didn't preserve any
    whitespace information.

    words (Iterable[str]): The words.
    text (str): The original text.
    RETURNS (Tuple[List[str], List[bool]]): The words and spaces.
    """
    if "".join("".join(words).split()) != "".join(text.split()):
        raise ValueError(Errors.E194.format(text=text, words=words))
    text_words = []
    text_spaces = []
    text_pos = 0
    # normalize words to remove all whitespace tokens
    norm_words = [word for word in words if not word.isspace()]
    # align words with text
    for word in norm_words:
        try:
            word_start = text[text_pos:].index(word)
        except ValueError:
            raise ValueError(Errors.E194.format(text=text, words=words)) from None
        if word_start > 0:
            text_words.append(text[text_pos : text_pos + word_start])
            text_spaces.append(False)
            text_pos += word_start
        text_words.append(word)
        text_spaces.append(False)
        text_pos += len(word)
        if text_pos < len(text) and text[text_pos] == " ":
            text_spaces[-1] = True
            text_pos += 1
    if text_pos < len(text):
        text_words.append(text[text_pos:])
        text_spaces.append(False)
    return (text_words, text_spaces)


def copy_config(config: Union[Dict[str, Any], Config]) -> Config:
    """Deep copy a Config. Will raise an error if the config contents are not
    JSON-serializable.

    config (Config): The config to copy.
    RETURNS (Config): The copied config.
    """
    try:
        return Config(config).copy()
    except ValueError:
        raise ValueError(Errors.E961.format(config=config)) from None


def dot_to_dict(values: Dict[str, Any]) -> Dict[str, dict]:
    """Convert dot notation to a dict. For example: {"token.pos": True,
    "token._.xyz": True} becomes {"token": {"pos": True, "_": {"xyz": True }}}.

    values (Dict[str, Any]): The key/value pairs to convert.
    RETURNS (Dict[str, dict]): The converted values.
    """
    result: Dict[str, dict] = {}
    for key, value in values.items():
        path = result
        parts = key.lower().split(".")
        for i, item in enumerate(parts):
            is_last = i == len(parts) - 1
            path = path.setdefault(item, value if is_last else {})
    return result


def dict_to_dot(obj: Dict[str, dict], *, for_overrides: bool = False) -> Dict[str, Any]:
    """Convert dot notation to a dict. For example: {"token": {"pos": True,
    "_": {"xyz": True }}} becomes {"token.pos": True, "token._.xyz": True}.

    obj (Dict[str, dict]): The dict to convert.
    for_overrides (bool): Whether to enable special handling for registered
        functions in overrides.
    RETURNS (Dict[str, Any]): The key/value pairs.
    """
    return {
        ".".join(key): value
        for key, value in walk_dict(obj, for_overrides=for_overrides)
    }


def dot_to_object(config: Config, section: str):
    """Convert dot notation of a "section" to a specific part of the Config.
    e.g. "training.optimizer" would return the Optimizer object.
    Throws an error if the section is not defined in this config.

    config (Config): The config.
    section (str): The dot notation of the section in the config.
    RETURNS: The object denoted by the section
    """
    component = config
    parts = section.split(".")
    for item in parts:
        try:
            component = component[item]
        except (KeyError, TypeError):
            raise KeyError(Errors.E952.format(name=section)) from None
    return component


def set_dot_to_object(config: Config, section: str, value: Any) -> None:
    """Update a config at a given position from a dot notation.

    config (Config): The config.
    section (str): The dot notation of the section in the config.
    value (Any): The value to set in the config.
    """
    component = config
    parts = section.split(".")
    for i, item in enumerate(parts):
        try:
            if i == len(parts) - 1:
                component[item] = value
            else:
                component = component[item]
        except (KeyError, TypeError):
            raise KeyError(Errors.E952.format(name=section)) from None


def walk_dict(
    node: Dict[str, Any], parent: List[str] = [], *, for_overrides: bool = False
) -> Iterator[Tuple[List[str], Any]]:
    """Walk a dict and yield the path and values of the leaves.

    for_overrides (bool): Whether to treat registered functions that start with
        @ as final values rather than dicts to traverse.
    """
    for key, value in node.items():
        key_parent = [*parent, key]
        if isinstance(value, dict) and (
            not for_overrides
            or not any(value_key.startswith("@") for value_key in value)
        ):
            yield from walk_dict(value, key_parent, for_overrides=for_overrides)
        else:
            yield (key_parent, value)


def get_arg_names(func: Callable) -> List[str]:
    """Get a list of all named arguments of a function (regular,
    keyword-only).

    func (Callable): The function
    RETURNS (List[str]): The argument names.
    """
    argspec = inspect.getfullargspec(func)
    return list(dict.fromkeys([*argspec.args, *argspec.kwonlyargs]))


def combine_score_weights(
    weights: List[Dict[str, Optional[float]]],
    overrides: Dict[str, Optional[float]] = SimpleFrozenDict(),
) -> Dict[str, Optional[float]]:
    """Combine and normalize score weights defined by components, e.g.
    {"ents_r": 0.2, "ents_p": 0.3, "ents_f": 0.5} and {"some_other_score": 1.0}.

    weights (List[dict]): The weights defined by the components.
    overrides (Dict[str, Optional[Union[float, int]]]): Existing scores that
        should be preserved.
    RETURNS (Dict[str, float]): The combined and normalized weights.
    """
    # We divide each weight by the total weight sum.
    # We first need to extract all None/null values for score weights that
    # shouldn't be shown in the table *or* be weighted
    result: Dict[str, Optional[float]] = {
        key: value for w_dict in weights for (key, value) in w_dict.items()
    }
    result.update(overrides)
    weight_sum = sum([v if v else 0.0 for v in result.values()])
    for key, value in result.items():
        if value and weight_sum > 0:
            result[key] = round(value / weight_sum, 2)
    return result


class DummyTokenizer:
    def __call__(self, text):
        raise NotImplementedError

    def pipe(self, texts, **kwargs):
        for text in texts:
            yield self(text)

    # add dummy methods for to_bytes, from_bytes, to_disk and from_disk to
    # allow serialization (see #1557)
    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, data: bytes, **kwargs) -> "DummyTokenizer":
        return self

    def to_disk(self, path: Union[str, Path], **kwargs) -> None:
        return None

    def from_disk(self, path: Union[str, Path], **kwargs) -> "DummyTokenizer":
        return self


def create_default_optimizer() -> Optimizer:
    return Adam()


def minibatch(items, size):
    """Iterate over batches of items. `size` may be an iterator,
    so that batch-size can vary on each step.
    """
    if isinstance(size, int):
        size_ = itertools.repeat(size)
    else:
        size_ = size
    items = iter(items)
    while True:
        batch_size = next(size_)
        batch = list(itertools.islice(items, int(batch_size)))
        if len(batch) == 0:
            break
        yield list(batch)


def is_cython_func(func: Callable) -> bool:
    """Slightly hacky check for whether a callable is implemented in Cython.
    Can be used to implement slightly different behaviors, especially around
    inspecting and parameter annotations. Note that this will only return True
    for actual cdef functions and methods, not regular Python functions defined
    in Python modules.

    func (Callable): The callable to check.
    RETURNS (bool): Whether the callable is Cython (probably).
    """
    attr = "__pyx_vtable__"
    if hasattr(func, attr):  # function or class instance
        return True
    # https://stackoverflow.com/a/55767059
    if (
        hasattr(func, "__qualname__")
        and hasattr(func, "__module__")
        and func.__module__ in sys.modules
    ):  # method
        cls_func = vars(sys.modules[func.__module__])[func.__qualname__.split(".")[0]]
        return hasattr(cls_func, attr)
    return False


def check_bool_env_var(env_var: str) -> bool:
    """Convert the value of an environment variable to a boolean. Add special
    check for "0" (falsy) and consider everything else truthy, except unset.

    env_var (str): The name of the environment variable to check.
    RETURNS (bool): Its boolean value.
    """
    value = os.environ.get(env_var, False)
    if value == "0":
        return False
    return bool(value)


def _pipe(
    docs: Iterable["Doc"],
    proc: "PipeCallable",
    name: str,
    default_error_handler: Callable[
        [str, "PipeCallable", List["Doc"], Exception], NoReturn
    ],
    kwargs: Mapping[str, Any],
) -> Iterator["Doc"]:
    if hasattr(proc, "pipe"):
        yield from proc.pipe(docs, **kwargs)
    else:
        # We added some args for pipe that __call__ doesn't expect.
        kwargs = dict(kwargs)
        error_handler = default_error_handler
        if hasattr(proc, "get_error_handler"):
            error_handler = proc.get_error_handler()
        for arg in ["batch_size"]:
            if arg in kwargs:
                kwargs.pop(arg)
        for doc in docs:
            try:
                doc = proc(doc, **kwargs)  # type: ignore[call-arg]
                yield doc
            except Exception as e:
                error_handler(name, proc, [doc], e)


def raise_error(proc_name, proc, docs, e):
    raise e


def ignore_error(proc_name, proc, docs, e):
    pass


def warn_if_jupyter_cupy():
    """Warn about require_gpu if a jupyter notebook + cupy + mismatched
    contextvars vs. thread ops are detected
    """
    if is_in_jupyter():
        from thinc.backends.cupy_ops import CupyOps

        if CupyOps.xp is not None:
            from thinc.backends import contextvars_eq_thread_ops

            if not contextvars_eq_thread_ops():
                warnings.warn(Warnings.W111)


def check_lexeme_norms(vocab, component_name):
    lexeme_norms = vocab.lookups.get_table("lexeme_norm", {})
    if len(lexeme_norms) == 0 and vocab.lang in LEXEME_NORM_LANGS:
        langs = ", ".join(LEXEME_NORM_LANGS)
        logger.debug(Warnings.W033.format(model=component_name, langs=langs))


def to_ternary_int(val) -> int:
    """Convert a value to the ternary 1/0/-1 int used for True/None/False in
    attributes such as SENT_START: True/1/1.0 is 1 (True), None/0/0.0 is 0
    (None), any other values are -1 (False).
    """
    if val is True:
        return 1
    elif val is None:
        return 0
    elif val is False:
        return -1
    elif val == 1:
        return 1
    elif val == 0:
        return 0
    else:
        return -1


# The following implementation of packages_distributions() is adapted from
# importlib_metadata, which is distributed under the Apache 2.0 License.
# Copyright (c) 2017-2019 Jason R. Coombs, Barry Warsaw
# See licenses/3rd_party_licenses.txt
def packages_distributions() -> Dict[str, List[str]]:
    """Return a mapping of top-level packages to their distributions. We're
    inlining this helper from the importlib_metadata "backport" here, since
    it's not available in the builtin importlib.metadata.
    """
    pkg_to_dist = defaultdict(list)
    for dist in importlib_metadata.distributions():
        for pkg in (dist.read_text("top_level.txt") or "").split():
            pkg_to_dist[pkg].append(dist.metadata["Name"])
    return dict(pkg_to_dist)


def all_equal(iterable):
    """Return True if all the elements are equal to each other
    (or if the input is an empty sequence), False otherwise."""
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)


def _is_port_in_use(port: int, host: str = "localhost") -> bool:
    """Check if 'host:port' is in use. Return True if it is, False otherwise.

    port (int): the port to check
    host (str): the host to check (default "localhost")
    RETURNS (bool): Whether 'host:port' is in use.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return False
    except socket.error:
        return True
    finally:
        s.close()


def find_available_port(start: int, host: str, auto_select: bool = False) -> int:
    """Given a starting port and a host, handle finding a port.

    If `auto_select` is False, a busy port will raise an error.

    If `auto_select` is True, the next free higher port will be used.

    start (int): the port to start looking from
    host (str): the host to find a port on
    auto_select (bool): whether to automatically select a new port if the given port is busy (default False)
    RETURNS (int): The port to use.
    """
    if not _is_port_in_use(start, host):
        return start

    port = start
    if not auto_select:
        raise ValueError(Errors.E1050.format(port=port))

    while _is_port_in_use(port, host) and port < 65535:
        port += 1

    if port == 65535 and _is_port_in_use(port, host):
        raise ValueError(Errors.E1049.format(host=host))

    # if we get here, the port changed
    warnings.warn(Warnings.W124.format(host=host, port=start, serve_port=port))
    return port
