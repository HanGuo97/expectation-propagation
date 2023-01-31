import jax
import functools
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation.baselines.cifar100 import image_classification_tasks as tff_cifar100_tasks
from tensorflow_federated.python.simulation.baselines.cifar100 import image_classification_preprocessing as tff_cifar100_preprocessing
from typing import List, Tuple, Union, Dict, Optional, Callable

from ..utils import misc_utils
from ..modules.utils import ModelIndex
from ..objectives import logistics_regression

# Ensure TF does not see GPU and grab all GPU memory.
# tf.config.set_visible_devices([], device_type="GPU")
NUM_POINTS_SLR_FILENAME = "/export/share/Experiments/20220801/num_points.slr.pth"


class DataHelper(object):

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def num_train_clients(self) -> int:
        raise NotImplementedError

    @property
    def centralized_train_objective(self) -> logistics_regression.SimpleObjective:
        raise NotImplementedError

    @property
    def centralized_test_objective(self) -> logistics_regression.SimpleObjective:
        raise NotImplementedError

    def get_client_train_objective(
        self,
        client_index: int,
    ) -> logistics_regression.SimpleObjective:
        raise NotImplementedError


class TFFDataHelper(DataHelper):

    def __init__(
        self,
        datasets: tff.simulation.baselines.BaselineTaskDatasets,
        model_index: ModelIndex,
        batch_size: int,
        num_epochs: int,
        num_classes: int,
        num_clients: Optional[int] = None,
    ) -> None:

        self._datasets = datasets
        self._model_index = model_index
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._num_classes = num_classes
        self._num_clients = num_clients

        self._centralized_train_objective = None
        self._centralized_test_objective = None

    @property
    def model_index(self) -> ModelIndex:
        return self._model_index

    @property
    def datasets(self) -> tff.simulation.baselines.BaselineTaskDatasets:
        return self._datasets

    @property
    def dim(self) -> int:
        return self.centralized_train_objective.dim

    @property
    def num_train_clients(self) -> int:
        if self._num_clients is not None:
            return self._num_clients
        return len(self.datasets._train_data.client_ids)

    @property
    def centralized_train_objective(self) -> logistics_regression.SimpleObjective:
        # Technically, since there could be randomization inside
        # the preprocessing, caching the first-preprocessed data
        # might not be the "best" way to do this. However,
        # this is fine for now to save time.
        if self._centralized_train_objective is None:
            if self._num_clients is not None:
                self._centralized_train_objective = (
                    self.get_centralized_train_objective(
                        list(range(self._num_clients))))
            else:
                self._centralized_train_objective = (
                    self.get_centralized_train_objective())

        return self._centralized_train_objective

    @property
    def centralized_test_objective(self) -> logistics_regression.SimpleObjective:
        if self._centralized_test_objective is None:
            self._centralized_test_objective = (
                self.get_centralized_test_objective())

        return self._centralized_test_objective

    @property
    def total_train_num_points(self) -> int:
        raise NotImplementedError

    def get_client_train_num_points(
            self,
            client_index: int,
    ) -> int:
        raise NotImplementedError

    def get_client_train_objective(
            self,
            client_index: int,
    ) -> logistics_regression.SimpleObjective:

        client_ids = (
            self
            .datasets
            ._preprocess_train_data
            .client_ids
        )

        tf_dataset = (
            self
            .datasets
            ._preprocess_train_data
            .create_tf_dataset_for_client(
                client_ids[client_index])
        )

        return self.create_objective_from_dataset(
            num_epochs=self._num_epochs,
            dataset=tf_dataset)

    def get_centralized_train_objective(
            self,
            client_indices: Optional[List[int]] = None,
    ) -> logistics_regression.SimpleObjective:
        # This uses train data, but preprocessed as evaluation
        # data. That is, deterministic preprocessing.
        # Note that the order of the data is not guaranteed.
        if not isinstance(self.datasets._train_data, tff.simulation.datasets.ClientData):
            raise ValueError
        if self.datasets._eval_preprocess_fn is None:
            raise ValueError

        if client_indices is None:
            tf_dataset_train = (
                self
                .datasets
                ._train_data
                .create_tf_dataset_from_all_clients())
        else:
            client_ids = (
                self
                .datasets
                ._train_data
                .client_ids
            )
            tf_dataset_train = (
                self
                .datasets
                ._train_data
                .create_tf_dataset_for_client(
                    client_ids[client_indices[0]])
            )
            for client_index in client_indices[1:]:
                _tf_dataset_train = (
                    self
                    .datasets
                    ._train_data
                    .create_tf_dataset_for_client(
                        client_ids[client_index])
                )

                tf_dataset_train = (
                    tf_dataset_train
                    .concatenate(_tf_dataset_train))

        tf_dataset_train = (
            self
            .datasets
            ._eval_preprocess_fn(
                tf_dataset_train))

        return self.create_objective_from_dataset(
            num_epochs=1,
            dataset=tf_dataset_train)

    def get_centralized_test_objective(
            self,
            client_indices: Optional[List[int]] = None,
    ) -> logistics_regression.SimpleObjective:
        # Note that the order of the data is not guaranteed.
        if not isinstance(self.datasets._test_data, tff.simulation.datasets.ClientData):
            raise ValueError
        if self.datasets._eval_preprocess_fn is None:
            raise ValueError

        if client_indices is None:
            tf_dataset_test = (
                self
                .datasets
                ._test_data
                .create_tf_dataset_from_all_clients())
        else:
            client_ids = (
                self
                .datasets
                ._test_data
                .client_ids
            )
            tf_dataset_test = (
                self
                .datasets
                ._test_data
                .create_tf_dataset_for_client(
                    client_ids[client_indices[0]])
            )
            for client_index in client_indices[1:]:
                _tf_dataset_test = (
                    self
                    .datasets
                    ._test_data
                    .create_tf_dataset_for_client(
                        client_ids[client_index])
                )

                tf_dataset_test = (
                    tf_dataset_test
                    .concatenate(_tf_dataset_test))

        tf_dataset_test = (
            self
            .datasets
            ._eval_preprocess_fn(
                tf_dataset_test))

        return self.create_objective_from_dataset(
            num_epochs=1,
            dataset=tf_dataset_test)

    def create_objective_from_dataset(
            self,
            num_epochs: int,
            dataset: tf.data.Dataset,
    ) -> logistics_regression.SimpleObjective:
        Xs = []
        Ys = []
        for X, Y in dataset.as_numpy_iterator():
            Xs.append(X)
            Ys.append(Y)

        return logistics_regression.SimpleObjective(
            model_index=self._model_index,
            Xs=jnp.concatenate(Xs, axis=0),
            Ys=jnp.concatenate(Ys, axis=0),
            batch_size=self._batch_size,
            num_epochs=num_epochs,
            num_classes=self._num_classes)

    @staticmethod
    def combine_objectives(
        objectives: List[logistics_regression.SimpleObjective],
    ) -> logistics_regression.SimpleObjective:
        checks = [
            (type(objective) == logistics_regression.SimpleObjective)
            for objective in objectives]
        if not all(checks):
            raise TypeError

        example_objective = objectives[-1]
        combined_Xs = jnp.concatenate(
            [objective.Xs for objective in objectives], axis=0)
        combined_Ys = jnp.concatenate(
            [objective.Ys for objective in objectives], axis=0)

        return logistics_regression.SimpleObjective(
            model_index=example_objective.model_index,
            Xs=combined_Xs,
            Ys=combined_Ys,
            batch_size=example_objective.batch_size,
            num_epochs=example_objective.num_epochs,
            num_classes=example_objective.num_classes,
        )

    @staticmethod
    def subsample_examples(
        objective: logistics_regression.SimpleObjective,
        size: int,
    ) -> logistics_regression.SimpleObjective:
        example_indices = np.random.choice(
            objective.num_points,
            size=size,
            replace=False).tolist()
        return logistics_regression.SimpleObjective(
            model_index=objective.model_index,
            Xs=objective.Xs[example_indices, ...],
            Ys=objective.Ys[example_indices, ...],
            batch_size=objective.batch_size,
            num_epochs=objective.num_epochs,
            num_classes=objective.num_classes)


class TFFCIFAR100(TFFDataHelper):

    def __init__(
        self,
        batch_size: int = 20,
        num_epochs: int = 10,
        num_clients: Optional[int] = None,
        v2: bool = True,
    ) -> None:

        train_client_spec = tff.simulation.baselines.ClientSpec(
            # Since we could have random preprocessing,
            # adding multiple epochs would effectively
            # mean data augmentation.
            num_epochs=num_epochs,
            batch_size=batch_size,
            # We do not shuffle data here, and assume
            # downstream code will shuffle data.
            shuffle_buffer_size=1,
        )
        eval_client_spec = tff.simulation.baselines.ClientSpec(
            num_epochs=1,
            batch_size=batch_size,
            shuffle_buffer_size=1,
        )

        task = tff.simulation.baselines.cifar100.create_image_classification_task(
            train_client_spec=train_client_spec,
            eval_client_spec=eval_client_spec,
        )

        if v2 is False:
            datasets = task.datasets
        else:
            crop_height = tff_cifar100_tasks.DEFAULT_CROP_HEIGHT
            crop_width = tff_cifar100_tasks. DEFAULT_CROP_WIDTH
            crop_shape = (crop_height, crop_width, 3)
            train_preprocess_fn = tff_cifar100_preprocessing.create_preprocess_fn(
                preprocess_spec=train_client_spec,
                crop_shape=crop_shape,
                distort_image=True)
            eval_preprocess_fn = tff_cifar100_preprocessing.create_preprocess_fn(
                preprocess_spec=eval_client_spec,
                crop_shape=crop_shape)

            datasets = tff.simulation.baselines.BaselineTaskDatasets(
                train_data=task.datasets._train_data,
                test_data=task.datasets._test_data,
                validation_data=task.datasets._validation_data,
                train_preprocess_fn=train_preprocess_fn,
                eval_preprocess_fn=eval_preprocess_fn)

        super().__init__(
            datasets=datasets,
            model_index=ModelIndex.CIFAR100,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_classes=100,
            num_clients=num_clients)

    @property
    def total_train_num_points(self) -> int:
        return 100 * self.num_train_clients

    def get_client_train_num_points(
            self,
            client_index: int,
    ) -> int:
        return 100


class TFFCIFAR100Toy(TFFCIFAR100):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model_index = ModelIndex.CIFAR100_TOY
        self._PCA = misc_utils.load(
            "/export/share/Experiments/20220903/"
            "pca.cifar100.dim100.cpkl")

    def create_objective_from_dataset(
            self, *args, **kwargs,
    ) -> logistics_regression.SimpleObjective:
        objective = super().create_objective_from_dataset(*args, **kwargs)
        return logistics_regression.SimpleObjective(
            model_index=objective.model_index,
            Xs=self._PCA.transform(objective.Xs.reshape(-1, 24 * 24 * 3)),
            Ys=objective.Ys,
            batch_size=objective.batch_size,
            num_epochs=objective.num_epochs,
            num_classes=objective.num_classes)


class TFFEMNIST62(TFFDataHelper):

    def __init__(
        self,
        batch_size: int = 20,
        num_epochs: int = 5,
        num_clients: Optional[int] = None,
    ) -> None:

        train_client_spec = tff.simulation.baselines.ClientSpec(
            # Since we could have random preprocessing,
            # adding multiple epochs would effectively
            # mean data augmentation.
            num_epochs=num_epochs,
            batch_size=batch_size,
            # We do not shuffle data here, and assume
            # downstream code will shuffle data.
            shuffle_buffer_size=1,
        )
        eval_client_spec = tff.simulation.baselines.ClientSpec(
            num_epochs=1,
            batch_size=batch_size,
            shuffle_buffer_size=1,
        )

        task = tff.simulation.baselines.emnist.create_character_recognition_task(
            train_client_spec=train_client_spec,
            eval_client_spec=eval_client_spec,
        )

        self._client_num_points_list = misc_utils.load(
            "/export/share/Experiments/20220517/"
            "client_num_points.emnist62.cpkl")

        super().__init__(
            datasets=task.datasets,
            model_index=ModelIndex.EMNIST62,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_classes=62,
            num_clients=num_clients)

    # https://stackoverflow.com/questions/4037481/caching-class-attributes-in-python
    # This requires creating centralized data, which could be expensive, so
    # we cache the output to save time.
    @functools.cached_property
    def dim(self) -> int:
        return super().dim

    @property
    def total_train_num_points(self) -> int:
        return sum([
            self.get_client_train_num_points(k)
            for k in range(self.num_train_clients)
        ])

    def get_client_train_num_points(
            self,
            client_index: int,
    ) -> int:
        return self._client_num_points_list[client_index]

    @property
    def centralized_train_objective(self) -> logistics_regression.SimpleObjective:
        # The training dataset is too large, so we
        # sample a small subset of it each time.
        client_indices = np.random.choice(
            self.num_train_clients,
            size=100, replace=False).tolist()
        return self.get_centralized_train_objective(client_indices)


class TFFStackOverflowLR(TFFDataHelper):
    TRAIN_MAX_ELEMENTS = 1000
    NUM_EVALUATION_EXAMPLES = 10000

    def __init__(
        self,
        batch_size: int = 100,
        num_epochs: int = 5,
        num_clients: Optional[int] = None,
    ) -> None:

        train_client_spec = tff.simulation.baselines.ClientSpec(
            # Google's implementaion uses `num_epochs=1` and repeat
            # multiple times. While we could uses `num_epochs=num_epochs`,
            # this gets a bit tricky when `max_elements` are used. Hence,
            # we follow a similar process by setting `num_epochs=1` and
            # manually repeating the dataset `num_epochs` times.
            num_epochs=1,
            batch_size=batch_size,
            max_elements=self.TRAIN_MAX_ELEMENTS,
            # We do not shuffle data here, and assume
            # downstream code will shuffle data.
            shuffle_buffer_size=1,
        )
        eval_client_spec = tff.simulation.baselines.ClientSpec(
            num_epochs=1,
            batch_size=batch_size,
            shuffle_buffer_size=1,
        )

        task = tff.simulation.baselines.stackoverflow.create_tag_prediction_task(
            train_client_spec=train_client_spec,
            eval_client_spec=eval_client_spec,
        )

        self._num_points_dict = misc_utils.load(NUM_POINTS_SLR_FILENAME)

        super().__init__(
            datasets=task.datasets,
            model_index=ModelIndex.STACKOVERFLOW_LR,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_classes=tff.simulation.baselines.stackoverflow.DEFAULT_TAG_VOCAB_SIZE,
            num_clients=num_clients)

    @property
    def total_train_num_points(self) -> int:
        df = self._num_points_dict["train"]
        df = df.apply(lambda value: min(value, self.TRAIN_MAX_ELEMENTS))
        return df.sum()

    def get_client_train_num_points(
            self,
            client_index: int,
    ) -> int:

        client_ids = (
            self
            .datasets
            ._preprocess_train_data
            .client_ids
        )

        client_id = client_ids[client_index]
        num_points = self._num_points_dict["train"].loc[client_id]
        # The num-points statistics are calculated before the
        # operation that takes up to `TRAIN_MAX_ELEMENTS` elements.
        return min(num_points, self.TRAIN_MAX_ELEMENTS)

    # https://stackoverflow.com/questions/4037481/caching-class-attributes-in-python
    # This requires creating centralized data, which could be expensive, so
    # we cache the output to save time.
    @functools.cached_property
    def dim(self) -> int:
        return super().dim

    def get_client_train_objective(
        self,
        client_index: int,
    ) -> logistics_regression.SimpleObjective:
        objectives = [
            # Make sure this gets updated when the class changes
            super(TFFStackOverflowLR, self)
            .get_client_train_objective(client_index)
            for _ in range(self._num_epochs)]
        return self.combine_objectives(objectives)

    def sample_clients_for_evaluation(self, split: str) -> List[int]:
        if split not in ["train", "test"]:
            raise ValueError
        
        if split == "train":
            client_ids = (
                self
                .datasets
                ._train_data
                .client_ids
            )
        if split == "test":
            client_ids = (
                self
                .datasets
                ._test_data
                .client_ids
            )

        client_indices = np.random.choice(
            len(client_ids),
            (len(client_ids),),
            replace=False)

        total_num_points = 0
        chosen_client_indices = []
        for k in client_indices:
            client_id = client_ids[k]
            chosen_client_indices.append(k)
            num_points = self._num_points_dict[split].loc[client_id]
            total_num_points = total_num_points + num_points
            if total_num_points > self.NUM_EVALUATION_EXAMPLES:
                break

        return chosen_client_indices

    @property
    def centralized_train_objective(self) -> logistics_regression.SimpleObjective:
        client_indices = self.sample_clients_for_evaluation("train")
        objective = self.get_centralized_train_objective(client_indices)
        return self.subsample_examples(objective, self.NUM_EVALUATION_EXAMPLES)

    @property
    def centralized_test_objective(self) -> logistics_regression.SimpleObjective:
        client_indices = self.sample_clients_for_evaluation("test")
        objective = self.get_centralized_test_objective(client_indices)
        return self.subsample_examples(objective, self.NUM_EVALUATION_EXAMPLES)


def count_client_num_points(
    data_helper: TFFDataHelper,
) -> Dict[str, List[str]]:
    processed_data_dicts = {}
    data_dicts = {
        "train": data_helper.datasets._train_data,
        "validation": data_helper.datasets._validation_data,
        "test": data_helper.datasets._test_data,
    }

    for key, data in data_dicts.items():
        _data = data._underlying_client_data
        print(f"{key:<10}\n\t {data} \n\t {_data}")

        query_parts = [
            f"SELECT client_id ",
            f"FROM examples ",
            f"WHERE split_name = '{_data._split_name}'"
        ]
        processed_data = tf.data.experimental.SqlDataset(
            driver_name="sqlite",
            data_source_name=_data._filepath,
            query=tf.strings.join(query_parts),
            output_types=(tf.string))
        processed_data_dicts[key] = list(processed_data.as_numpy_iterator())

    return processed_data_dicts
