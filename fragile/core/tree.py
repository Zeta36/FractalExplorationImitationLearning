import copy
from typing import Any, List, Set

import networkx as nx

from fragile.core.base_classes import BaseStateTree
from fragile.core.states import StatesEnv, StatesModel, StatesWalkers


class _BaseNetworkxTree(BaseStateTree):
    """
    This is a tree data structure that stores the paths followed by the walkers. \
    It can be pruned to delete paths that are longer be needed. It uses a \
    networkx DiGraph to keep track of the states relationships.
    """

    def __init__(self):
        """Initialize a :class:`_BaseNetworkxTree`."""
        self.data: nx.DiGraph = nx.DiGraph()
        self.data.add_node(self.ROOT_ID, state=None, n_iter=-1)
        self.root_id = self.ROOT_ID
        self.ids_to_hash = {self.ROOT_ID: self.ROOT_HASH}
        self.hash_to_ids = {self.ROOT_HASH: self.ROOT_ID}
        self._node_count = 0
        self.leafs = {self.ROOT_ID}
        self.parents = {self.ROOT_ID}

    def reset(
        self,
        env_states: StatesEnv = None,
        model_states: StatesModel = None,
        walkers_states: StatesWalkers = None,
    ) -> None:
        """
        Delete all the data currently stored and reset the internal state of \
        the tree .

        Args:
            env_states: Ignored. Only to implement interface.
            model_states: Ignored. Only to implement interface.
            walkers_states: Ignored. Only to implement interface.

        Returns:
            None.

        """
        self.data: nx.DiGraph = nx.DiGraph()
        self.data.add_node(self.ROOT_ID, state=None, n_iter=-1)
        self.root_id = self.ROOT_ID
        self.ids_to_hash = {self.ROOT_ID: self.ROOT_HASH}
        self.hash_to_ids = {self.ROOT_HASH: self.ROOT_ID}
        self._node_count = 0
        self.leafs = {self.ROOT_ID}
        self.parents = {self.ROOT_ID}

    def add_new_hash(self, node_hash: int) -> int:
        """
        Generate and assign a node id to a new node hash and return the node_id \
        assigned to it.
        """
        if node_hash in self.hash_to_ids:
            raise ValueError("Hash is already present with id %s " % self.ids_to_hash[node_hash])
        self._node_count += 1
        node_id = int(self._node_count)
        self.ids_to_hash[node_id] = node_hash
        self.hash_to_ids[node_hash] = node_id
        return node_id

    def append_leaf(
        self,
        leaf_id: int,
        parent_id: int,
        state: Any,
        action: Any = 1,
        dt: int = None,
        n_iter: int = None,
        from_hash: bool = False,
        reward: float = 0.0,
        cum_reward: float = 0.0,
    ):
        """
        Add a new state as a leaf node of the tree to keep track of the \
        trajectories of the swarm.

        Args:
            leaf_id: Id that identifies the state that will be added to the tree. \
                     If the id is the hash of a walker state ``from_hash`` needs \
                     to be ``True``. Otherwise it refers to a node_id the node \
                     will be assigned.
            parent_id: Id that identifies the state of the node parent that will \
                       be added to the tree. If the id is the hash of a walker state \
                       ``from_hash`` needs to be ``True``. Otherwise it refers to \
                       the parent node_id.
            state: observation assigned to ``leaf_id`` node.
            action: action taken at ``leaf_id`` node.
            dt: Number of steps taken from the parent node to ``leaf_id``.
            n_iter: Swarm iteration when the current node was generated.
            from_hash: If  ``True`` ``node_id`` and ``parent_id`` will be \
                      considered hashes of states. If ``False`` they will be \
                      considered node ids.
            reward: Instantaneous reward assigned to the node that will be added.
            cum_reward: Cumulative reward assigned to the node that will be added.

        Returns:
            None.

        """
        leaf_name = (
            (
                self.add_new_hash(leaf_id)
                if leaf_id not in self.hash_to_ids
                else self.hash_to_ids[leaf_id]
            )
            if from_hash
            else leaf_id
        )

        parent_name = self.hash_to_ids[parent_id] if from_hash else parent_id
        if leaf_name not in self.data.nodes and leaf_name != parent_name:
            self.data.add_node(
                leaf_name, state=state, n_iter=n_iter, reward=reward, cum_reward=cum_reward
            )
            self.data.add_edge(parent_name, leaf_name, action=action, dt=dt)
            self.leafs.add(leaf_name)
            if parent_name in self.leafs:
                self.leafs.remove(parent_name)

    def prune_tree(self, dead_leafs: Set[int], alive_leafs: Set[int], from_hash: bool = False):
        """
        Prune the orphan leaves that will no longer be used in order to save memory.

        Args:
            dead_leafs: Leaves of the branches that will be removed.
            alive_leafs: Leaves of the branches that will kept being expanded.
            from_hash: If ``True`` ``dead_leafs`` and ``alive_leafs`` will be \
                      considered hashes of states. If ``False`` they will be \
                      considered node ids.

        Returns:
            None

        """
        for leaf in dead_leafs:
            self.prune_branch(leaf, alive_leafs, from_hash=from_hash)
        return

    def get_branch(self, leaf_id, from_hash: bool = False, root=BaseStateTree.ROOT_ID) -> tuple:
        """
        Get the data of the branch ended at leaf_id.

        Args:
            leaf_id: Id that identifies the leaf of the tree. \
                     If ``leaf_id`` is the hash of a walker state ``from_hash`` \
                     needs to be ``True``. Otherwise it refers to a node id of \
                     the leaf node.
            from_hash: If  ``True`` ``leaf_id`` is considered the hash of walker \
                      state. If ``False`` it will be considered a node id.
            root: Node id of the root node of the tree.

        Returns:
            tuple containing (states, actions, dts) that represent the history \
            of a given branch of the tree.

            ``states`` represent the :class:`StatesEnv`.states assigned to each node.
            ``actions`` represent the :class:`StatesModel`.actions taken at each state.
            ``dts`` the :class:`StatesModel`.dt of each state.

        """
        leaf_name = self.hash_to_ids[leaf_id] if from_hash else leaf_id
        nodes = nx.shortest_path(self.data, root, leaf_name)
        states = [self.data.nodes[n]["state"] for n in nodes]
        actions = [self.data.edges[(n, nodes[i + 1])]["action"] for i, n in enumerate(nodes[:-1])]
        dts = [self.data.edges[(n, nodes[i + 1])]["dt"] for i, n in enumerate(nodes[:-1])]
        return states, actions, dts

    def prune_branch(self, leaf_id: int, alive_leafs: set, from_hash: bool = False):
        """
        Recursively prunes a branch that ends in an orphan leaf.

        Args:
            leaf_id: Id that identifies the leaf of the tree. \
                     If ``leaf_id`` is the hash of a walker state ``from_hash`` \
                     needs to be ``True``. Otherwise it refers to a node id of \
                     the leaf node.
            alive_leafs: Leaves of the branches that will kept being expanded.
            from_hash: If ``True`` ``leaf_id`` and ``alive_leafs`` will be \
                      considered hashes of states. If ``False`` they will be \
                      considered node ids.

        Returns:
            None

        """
        leaf = self.hash_to_ids[leaf_id] if from_hash else leaf_id
        is_not_a_leaf = len(self.data.out_edges([leaf])) > 0

        if (
            is_not_a_leaf
            or leaf == self.ROOT_ID
            or leaf not in self.data.nodes
            or leaf in self.parents
        ):
            return
        alive_leafs = (
            set([self.hash_to_ids[le] for le in alive_leafs]) if from_hash else set(alive_leafs)
        )
        if leaf in alive_leafs:
            return
        # Remove the node if it is a leaf and is not alive
        parents = list(self.data.in_edges([leaf]))
        parent = parents[0][0]
        if parent == self.ROOT_ID or parent in alive_leafs or parent in self.parents:
            return
        self.data.remove_node(leaf)
        self.leafs.discard(leaf)
        self.leafs.add(parent)
        leaf_hash = self.ids_to_hash[leaf]
        del self.ids_to_hash[leaf]
        del self.hash_to_ids[leaf_hash]
        return self.prune_branch(parent, alive_leafs)

    def get_parent(self, node_id, from_hash: bool = False) -> int:
        """Get the node id of the parent of the target node."""
        node_id = self.hash_to_ids[node_id] if from_hash else node_id
        return list(self.data.in_edges(node_id))[0][0]

    def get_leaf_nodes(self) -> List[int]:
        """Return a list containing all the node ids of the leaves of the tree."""
        leafs = []
        for node in self.data.nodes:
            if len(self.data.out_edges([node])) == 0:
                leafs.append(node)
        return leafs


class HistoryTree(_BaseNetworkxTree):
    """Keep track of the history of trajectories generated bu the :class:`Swarm`."""

    def add_states(
        self,
        parent_ids: List[int],
        env_states: StatesEnv = None,
        model_states: StatesModel = None,
        walkers_states: StatesWalkers = None,
        n_iter: int = None,
    ):
        """
        Update the history of the tree adding the necessary data to recreate a \
        the trajectories sampled by the :class:`Swarm`.

        Args:
            parent_ids: List of states hashes representing the parent nodes of \
                        the current states.
            env_states: :class:`StatesEnv` containing the data that will be \
                        saved as new leaves in the tree.
            model_states: :class:`StatesModel` containing the data that will be \
                        saved as new leaves in the tree.
            walkers_states: :class:`StatesWalkers` containing the data that will be \
                        saved as new leaves in the tree.
            n_iter: Number of iteration of the algorithm when the data was sampled.

        Returns:
            None

        """
        leaf_ids = walkers_states.id_walkers.tolist()
        self.parents = set(self.hash_to_ids[pa] for pa in parent_ids)
        for i, (leaf, parent) in enumerate(zip(leaf_ids, parent_ids)):
            state = copy.deepcopy(env_states.states[i])
            reward = copy.deepcopy(env_states.rewards[i])
            cum_reward = copy.deepcopy(walkers_states.cum_rewards[i])
            action = copy.deepcopy(model_states.actions[i])
            dt = copy.copy(model_states.dt[i])
            self.append_leaf(
                leaf,
                parent,
                state,
                action,
                dt,
                n_iter=n_iter,
                from_hash=True,
                reward=reward,
                cum_reward=cum_reward,
            )

    def prune_tree(self, alive_leafs: Set[int], from_hash: bool = False):
        """
        Remove the branches that do not have a walker in their leaves.

        Args:
            alive_leafs: Contains the ids  of the leaf nodes that are being \
                         expanded by the walkers.
            from_hash: from_hash: If ``True`` ``alive_leafs`` will be \
                      considered a set of hashes of states. If ``False`` it \
                      will be considered a set of node ids.

        Returns:
            None.

        """
        alive_leafs = set([self.hash_to_ids[le] if from_hash else le for le in set(alive_leafs)])
        dead_leafs = self.leafs - alive_leafs
        super(HistoryTree, self).prune_tree(
            dead_leafs=dead_leafs, alive_leafs=alive_leafs, from_hash=False
        )
