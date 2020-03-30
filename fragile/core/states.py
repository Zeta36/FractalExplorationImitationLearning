import copy
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

import numpy

from fragile.core.utils import float_type, hash_numpy, Scalar, similiar_chunks_indexes, StateDict


class States:
    """
    Handles several tensors that will contain the data associated with the \
    walkers of a Swarm. Each tensor will be associated to a class attribute.

    This class behaves as a dictionary of tensors with some extra functionality
    to make easier the process of cloning the walkers' states by adding an \
    extra dimension equal to the number of walkers to each tensor.

    In order to define the tensors, a `state_dict` dictionary needs to be \
    specified using the following structure::

        state_dict = {"name_1": {"size": tuple([1]),
                                 "dtype": numpy.float32,
                                },
                     }

    Where tuple is a tuple indicating the shape of the desired tensor, that will \
    be accessed using the name_1 attribute of the class. If "size" is not defined \
    the attribute will be considered a vector of length `batch_size`.


    Args:
        batch_size: The number of items in the first dimension of the tensors.
        state_dict: Dictionary defining the attributes of the tensors.
        **kwargs: Data can be directly specified as keyword arguments.
    """

    def __init__(self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs):
        """
        Initialise a :class:`States`.

        Args:
             batch_size: The number of items in the first dimension of the tensors.
             state_dict: Dictionary defining the attributes of the tensors.
             **kwargs: The name-tensor pairs can also be specified as kwargs.

        """
        attr_dict = self.params_to_arrays(state_dict, batch_size) if state_dict is not None else {}
        attr_dict.update(kwargs)
        self._names = list(attr_dict.keys())
        self._attr_dict = attr_dict
        self.update(**self._attr_dict)
        self._n_walkers = batch_size

    def __len__(self):
        """Length is equal to n_walkers."""
        return self.n

    def __getitem__(self, item: str) -> Union[numpy.ndarray, List[numpy.ndarray]]:
        """
        Query an attribute of the class as if it was a dictionary.

        Args:
            item: Name of the attribute to be selected.

        Returns:
            The corresponding item.

        """
        if isinstance(item, str):
            try:
                return getattr(self, item)
            except AttributeError:
                raise TypeError("Tried to get a non existing attribute with key {}".format(item))
        else:
            raise TypeError("item must be an instance of str, got {} instead".format(item))

    def __setitem__(self, key, value: Union[Tuple, List, numpy.ndarray]):
        """
        Allow the class to set its attributes as if it was a dict.

        Args:
            key: Attribute to be set.
            value: Value of the target attribute.

        Returns:
            None.

        """
        if key not in self._names:
            self._names.append(key)
        self.update(**{key: value})

    def __repr__(self):
        string = "{} with {} walkers\n".format(self.__class__.__name__, self.n)
        for k, v in self.items():
            shape = v.shape if hasattr(v, "shape") else None
            new_str = "{}: {} {}\n".format(k, type(v), shape)
            string += new_str
        return string

    def __hash__(self) -> int:
        _hash = hash(
            tuple(
                [hash_numpy(x) if isinstance(x, numpy.ndarray) else hash(x) for x in self.vals()]
            )
        )
        return _hash

    def group_hash(self, name: str) -> int:
        """Return a unique id for a given attribute."""
        val = getattr(self, name)
        return hash_numpy(val) if isinstance(val, numpy.ndarray) else hash(val)

    def hash_values(self, name: str) -> List[int]:
        """Return a unique id for each walker attribute."""
        values = getattr(self, name)
        hashes = [
            hash_numpy(val) if isinstance(val, numpy.ndarray) else hash(val) for val in values
        ]
        return hashes

    @staticmethod
    def merge_states(states: Iterable["States"]) -> "States":
        """
        Combine different states containing the same kind of data into a single \
        :class:`State` with batch size equal to the sum of all the state batch \
        sizes.

        Args:
            states: Iterable returning :class:`State`s with the same attributes.

        Returns:
            :class:`State` containing the combined data of the input values.

        """

        def merge_one_name(states_list, name):
            vals = []
            is_scalar_vector = True
            for state in states_list:
                data = state[name]
                # Attributes that are not numpy arrays are not stacked.
                if not isinstance(data, numpy.ndarray):
                    return data
                state_len = len(state)
                if len(data.shape) == 0 and state_len == 1:
                    # Name is scalar vector. Data is scalar value. Transform to array first
                    value = numpy.array([data]).flatten()
                elif len(data.shape) == 1 and state_len == 1:
                    if data.shape[0] == 1:
                        # Name is scalar vector. Data already transformed to an array
                        value = data
                    else:
                        # Name is a matrix of vectors. Data needs an additional dimension
                        is_scalar_vector = False
                        value = numpy.array([data])
                elif len(data.shape) == 1 and state_len > 1:
                    # Name is a scalar vector. Data already has is a one dimensional array
                    value = data
                elif (
                    len(data.shape) > 1
                    and state_len > 1
                    or len(data.shape) > 1
                    and len(state) == 1
                ):
                    # Name is a matrix of vectors. Data has the correct shape
                    is_scalar_vector = False
                    value = data
                else:
                    raise ValueError(
                        "Could not infer data concatenation for attribute %s  with shape %s"
                        % (name, data.shape)
                    )
                vals.append(value)
            if is_scalar_vector:
                return numpy.concatenate(vals)
            else:
                return numpy.vstack(vals)

        # Assumes all states have the same names.
        data = {name: merge_one_name(states, name) for name in states[0]._names}
        batch_size = sum(s.n for s in states)
        return states[0].__class__(batch_size=batch_size, **data)

    @property
    def n(self) -> int:
        """Return the batch_size of the vectors, which is equivalent to the number of walkers."""
        return self._n_walkers

    def get(self, key: str, default=None):
        """
        Get an attribute by key and return the default value if it does not exist.

        Args:
            key: Attribute to be recovered.
            default: Value returned in case the attribute is not part of state.

        Returns:
            Target attribute if found in the instance, otherwise returns the
             default value.

        """
        if key not in self.keys():
            return default
        return self[key]

    def keys(self) -> Generator:
        """Return a generator for the attribute names of the stored data."""
        return (n for n in self._names)

    def vals(self) -> Generator:
        """Return a generator for the values of the stored data."""
        return (self[name] for name in self._names)

    def items(self) -> Generator:
        """Return a generator for the attribute names and the values of the stored data."""
        return ((name, self[name]) for name in self._names)

    def itervals(self):
        """
        Iterate the states attributes by walker.

        Returns:
            Tuple containing all the names of the attributes, and the values that
            correspond to a given walker.

        """
        if self.n <= 1:
            return self.vals()
        for i in range(self.n):
            yield tuple([v[i] for v in self.vals()])

    def iteritems(self):
        """
        Iterate the states attributes by walker.

        Returns:
            Tuple containing all the names of the attributes, and the values that
            correspond to a given walker.

        """
        if self.n < 1:
            return self.vals()
        for i in range(self.n):
            values = (v[i] if isinstance(v, numpy.ndarray) else v for v in self.vals())
            yield tuple(self._names), tuple(values)

    def split_states(self, n_chunks: int) -> Generator["States", None, None]:
        """
        Return a generator for n_chunks different states, where each one \
        contain only the data corresponding to one walker.
        """

        def get_chunck_size(state, start, end):
            for name in state._names:
                attr = state[name]
                if isinstance(attr, numpy.ndarray):
                    return len(attr[start:end])
            return int(numpy.ceil(self.n / n_chunks))

        for start, end in similiar_chunks_indexes(self.n, n_chunks):
            chunk_size = get_chunck_size(self, start, end)

            data = {
                k: val[start:end] if isinstance(val, numpy.ndarray) else val
                for k, val in self.items()
            }
            new_state = self.__class__(batch_size=chunk_size, **data)
            yield new_state

    def update(self, other: "States" = None, **kwargs):
        """
        Modify the data stored in the States instance.

        Existing attributes will be updated, and new attributes will be created if needed.

        Args:
            other: State class that will be copied upon update.
            **kwargs: It is possible to specify the update as key value attributes, \
                     where key is the name of the attribute to be updated, and value \
                      is the new value for the attribute.
        """

        def update_or_set_attributes(attrs: Union[dict, States]):            
            for name, val in attrs.items():            
                try:
                    getattr(self, name)[:] = copy.deepcopy(val)
                except (AttributeError, TypeError, KeyError, ValueError):
                    setattr(self, name, copy.deepcopy(val))                             

        if other is not None:
            update_or_set_attributes(other)
        if kwargs:
            update_or_set_attributes(kwargs)

    def clone(
        self,
        will_clone: numpy.ndarray,
        compas_ix: numpy.ndarray,
        ignore: Optional[Set[str]] = None,
    ):
        """
        Clone all the stored data according to the provided arrays.

        Args:
            will_clone: Array of shape (n_walkers,) of booleans indicating the \
                        index of the walkers that will clone to a random companion.
            compas_ix: Array of integers of shape (n_walkers,). Contains the \
                       indexes of the walkers that will be copied.
            ignore: set containing the names of the attributes that will not be \
                    cloned.

        """
        ignore = set() if ignore is None else ignore
        for name in self.keys():
            if isinstance(self[name], numpy.ndarray) and name not in ignore:
                self[name][will_clone] = self[name][compas_ix][will_clone]

    def get_params_dict(self) -> StateDict:
        """Return a dictionary describing the data stored in the :class:`States`."""
        return {
            k: {"shape": v.shape, "dtype": v.dtype}
            for k, v in self.__dict__.items()
            if isinstance(v, numpy.ndarray)
        }

    def copy(self) -> "States":
        """Crete a copy of the current instance."""
        param_dict = {str(name): val.copy() for name, val in self.items()}
        return States(batch_size=self.n, **param_dict)

    @staticmethod
    def params_to_arrays(param_dict: StateDict, n_walkers: int) -> Dict[str, numpy.ndarray]:
        """
        Create a dictionary containing the arrays specified by param_dict.

        Args:
            param_dict: Dictionary defining the attributes of the tensors.
            n_walkers: Number items in the first dimension of the data tensors.

        Returns:
              Dictionary with the same keys as param_dict, containing arrays specified \
              by `param_dict` values.

        """
        tensor_dict = {}
        for key, val in param_dict.items():
            # Shape already includes the number of walkers. Remove walkers axis to create size.
            shape = val.get("shape")
            if shape is None:
                val_size = val.get("size")
            elif len(shape) > 1:
                val_size = shape[1:]
            else:
                val_size = val.get("size")
            # Create appropriate shapes with current state's number of walkers.
            sizes = n_walkers if val_size is None else tuple([n_walkers]) + val_size
            if "size" in val:
                del val["size"]
            if "shape" in val:
                del val["shape"]
            tensor_dict[key] = numpy.zeros(shape=sizes, **val)
        return tensor_dict


class StatesEnv(States):
    """Keeps track of the data structures used by the :class:`Env`."""

    def __init__(self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs):
        """
        Initialise a :class:`StatesEnv`.

        Args:
             batch_size: The number of items in the first dimension of the tensors.
             state_dict: Dictionary defining the attributes of the tensors.
             **kwargs: The name-tensor pairs can also be specified as kwargs.

        """
        self.observs = None
        self.states = None
        self.rewards = None
        self.ends = None
        updated_dict = self.get_params_dict()
        if state_dict is not None:
            updated_dict.update(state_dict)
        super(StatesEnv, self).__init__(state_dict=updated_dict, batch_size=batch_size, **kwargs)

    def get_params_dict(self) -> StateDict:
        """Return a dictionary describing the data stored in the :class:`StatesEnv`."""
        params = {
            "states": {"dtype": numpy.int64},
            "observs": {"dtype": numpy.float32},
            "rewards": {"dtype": numpy.float32},
            "ends": {"dtype": numpy.bool_},
        }
        state_dict = super(StatesEnv, self).get_params_dict()
        params.update(state_dict)
        return params


class StatesModel(States):
    """Keeps track of the data structures used by the :class:`Model`."""

    def __init__(self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs):
        """
        Initialise a :class:`StatesModel`.

        Args:
             batch_size: The number of items in the first dimension of the tensors.
             state_dict: Dictionary defining the attributes of the tensors.
             **kwargs: The name-tensor pairs can also be specified as kwargs.

        """
        self.actions = None
        updated_dict = self.get_params_dict()
        if state_dict is not None:
            updated_dict.update(state_dict)
        super(StatesModel, self).__init__(state_dict=updated_dict, batch_size=batch_size, **kwargs)

    def get_params_dict(self) -> StateDict:
        """Return the parameter dictionary with tre attributes common to all Models."""
        params = {
            "actions": {"dtype": numpy.float32},
        }
        state_dict = super(StatesModel, self).get_params_dict()
        params.update(state_dict)
        return params


class StatesWalkers(States):
    """Keeps track of the data structures used by the :class:`Walkers`."""

    def __init__(self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs):
        """
        Initialize a :class:`StatesWalkers`.

        Args:
            batch_size: Number of walkers that the class will be tracking.
            state_dict: Dictionary defining the attributes of the tensors.
            kwargs: attributes that will not be set as numpy.ndarrays
        """
        self.will_clone = None
        self.compas_clone = None
        self.processed_rewards = None
        self.cum_rewards = None
        self.virtual_rewards = None
        self.distances = None
        self.clone_probs = None
        self.alive_mask = None
        self.id_walkers = None
        self.end_condition = None
        updated_dict = self.get_params_dict()
        if state_dict is not None:
            updated_dict.update(state_dict)
        super(StatesWalkers, self).__init__(
            state_dict=updated_dict, batch_size=batch_size, **kwargs
        )
        self.best_id = 0
        self.best_obs = None
        self.best_state = None
        # This is only to allow __repr__. Should be overriden after reset
        self.best_reward = -numpy.inf

    @property
    def best_found(self) -> numpy.ndarray:
        """Return the best observation found."""
        return self.best_obs

    @property
    def best_reward_found(self) -> Scalar:
        """Return the value of the best observation found."""
        return self.best_reward

    def get_params_dict(self) -> StateDict:
        """Return a dictionary containing the param_dict to build an instance \
        of States that can handle all the data generated by the :class:`Walkers`.
        """
        params = {
            "id_walkers": {"dtype": numpy.int64},
            "compas_clone": {"dtype": numpy.int64},
            "processed_rewards": {"dtype": float_type},
            "virtual_rewards": {"dtype": float_type},
            "cum_rewards": {"dtype": float_type},
            "distances": {"dtype": float_type},
            "clone_probs": {"dtype": float_type},
            "will_clone": {"dtype": numpy.bool_},
            "alive_mask": {"dtype": numpy.bool_},
            "end_condition": {"dtype": numpy.bool_},
        }
        state_dict = super(StatesWalkers, self).get_params_dict()
        params.update(state_dict)
        return params

    def clone(self, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Perform the clone only on cum_rewards and id_walkers and reset the other arrays."""
        clone, compas = self.will_clone, self.compas_clone
        self.cum_rewards[clone] = copy.deepcopy(self.cum_rewards[compas][clone])
        self.id_walkers[clone] = copy.deepcopy(self.id_walkers[compas][clone])
        return clone, compas

    def reset(self):
        """Clear the internal data of the class."""
        other_attrs = [name for name in self.keys() if name not in self.get_params_dict()]
        for attr in other_attrs:
            setattr(self, attr, None)
        self.update(
            id_walkers=numpy.zeros(self.n, dtype=numpy.int64),
            compas_dist=numpy.arange(self.n),
            compas_clone=numpy.arange(self.n),
            processed_rewards=numpy.zeros(self.n, dtype=float_type),
            cum_rewards=numpy.zeros(self.n, dtype=float_type),
            virtual_rewards=numpy.ones(self.n, dtype=float_type),
            distances=numpy.zeros(self.n, dtype=float_type),
            clone_probs=numpy.zeros(self.n, dtype=float_type),
            will_clone=numpy.zeros(self.n, dtype=numpy.bool_),
            alive_mask=numpy.ones(self.n, dtype=numpy.bool_),
            end_condition=numpy.zeros(self.n, dtype=numpy.bool_),
        )
