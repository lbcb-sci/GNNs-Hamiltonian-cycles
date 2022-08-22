import os

from src.constants import DEFAULT_DATASET_SIZES
from src.Evaluation import EvaluationScores

EVALUATION_DATABASE_PATH = os.path.join("EVALUATIONS", "Evaluation_results.csv")
EVALUATION_WEIGHTS_BACKUP_PATH = os.path.join("EVALUATIONS", "WEIGHTS_BACKUP")
EVALUATION_SIZES = DEFAULT_DATASET_SIZES


class EVALUATION_TABLE_TAGS:
    id = "id"
    model_name = "model_name"
    train_epochs = "train_epochs"
    iterations_in_epoch = "iterations_in_epoch"
    final_loss_exp = "final_loss_exp"
    final_loss_var = "final_loss_var"
    final_loss_trend = "final_loss_trend"
    train_graphs_seen = "train_graphs_seen"
    train_description = "train_description"
    train_examples_details = "train_example_details"
    learn_rate = "learn_rate"
    optimizer_name = "optimizer_name"
    sample_method = "sample_method"
    loss_name = "loss_name"
    loss_description = "loss_description"
    weights_path = "weights_path"
    description = "description"
    date = "date"
    evaluation_data = "evaluation_data"
    raw_model_details = "model_raw"
    raw_optimizer_details = "optimizer_raw"
    computed_by = "computed_by"
    train_time = "train_time(s)"
    batch_size = "batch_size"


def create_archive_evaluation_entry(nn_hamilton, generator, model_name, history, iterations_in_epoch, learn_rate, optimizer,
                                    loss_name, loss_description=None, description=None, training_graphs_seen=None,
                                    sample_method=None, train_description=None, batch_size=None):
    import time
    import datetime
    import socket
    import threading
    data_files = str.join(",", [os.path.basename(p) for p in os.listdir("DATA") if os.path.splitext(p)[1] == ".pt"])
    id_str = str(time.time_ns()) + f"#{threading.get_ident()}"
    weights_directory = os.path.join(EVALUATION_WEIGHTS_BACKUP_PATH, id_str)
    os.mkdir(weights_directory)
    nn_hamilton.save_weights(weights_directory)
    epochs = len(history["epoch_avg"])
    assert epochs > 0
    _smoothing_window = 100
    if epochs >= _smoothing_window:
        loss_exp = sum(history["epoch_avg"][-_smoothing_window:]) / _smoothing_window
        loss_var = sum([x**2 for x in history["epoch_avg"][_smoothing_window:]]) / _smoothing_window - loss_exp**2
        loss_trend = (sum(history["epoch_avg"][-_smoothing_window // 2:])
                     - sum(history["epoch_avg"][-_smoothing_window: -_smoothing_window // 2])) / _smoothing_window
    else:
        loss_exp, loss_var, loss_trend = [None]*3

    item_dict = {
        EVALUATION_TABLE_TAGS.id: id_str,
        EVALUATION_TABLE_TAGS.model_name: model_name,
        EVALUATION_TABLE_TAGS.description: description,
        EVALUATION_TABLE_TAGS.weights_path: weights_directory,
        EVALUATION_TABLE_TAGS.iterations_in_epoch: iterations_in_epoch,
        EVALUATION_TABLE_TAGS.train_epochs: epochs,
        EVALUATION_TABLE_TAGS.train_graphs_seen: training_graphs_seen,
        EVALUATION_TABLE_TAGS.train_examples_details: generator.output_details(),
        EVALUATION_TABLE_TAGS.train_description: train_description,
        EVALUATION_TABLE_TAGS.sample_method: sample_method,
        EVALUATION_TABLE_TAGS.learn_rate: learn_rate,
        EVALUATION_TABLE_TAGS.optimizer_name: optimizer.__class__.__name__,
        EVALUATION_TABLE_TAGS.raw_optimizer_details: str(optimizer),
        EVALUATION_TABLE_TAGS.raw_model_details: str(nn_hamilton.description()),
        EVALUATION_TABLE_TAGS.loss_name: loss_name,
        EVALUATION_TABLE_TAGS.loss_description: loss_description,
        EVALUATION_TABLE_TAGS.date: datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        EVALUATION_TABLE_TAGS.evaluation_data: data_files,
        EVALUATION_TABLE_TAGS.computed_by: socket.gethostname(),
        EVALUATION_TABLE_TAGS.final_loss_exp: loss_exp,
        EVALUATION_TABLE_TAGS.final_loss_var: loss_var,
        EVALUATION_TABLE_TAGS.final_loss_trend: loss_trend,
        EVALUATION_TABLE_TAGS.train_time: history["train_time"],
        EVALUATION_TABLE_TAGS.batch_size: batch_size
    }
    missing_tags = [v for k, v in vars(EVALUATION_TABLE_TAGS).items() if not k.startswith("_") and v not in item_dict]
    if len(missing_tags) > 0:
        print("Missing tags {} while logging the model! Please edit these manually.".format(missing_tags))

    _test_success_flag = False
    _test_batches_per_size = 10
    print("Testing {} for basic functionality...".format(nn_hamilton.__class__.__name__))
    evals, sizes = EvaluationScores.evaluate_on_saved_data(nn_hamilton, _test_batches_per_size, ["DATA"])
    acc_scores = EvaluationScores.compute_accuracy_scores(evals, sizes)
    if any([acc[i] > 0 for acc in acc_scores[:4] for i in range(len(sizes))]):
        _test_success_flag = True
    if not _test_success_flag:
        print(f"{nn_hamilton.__class__.__name__} does not produce a single non-zero score. Aborting.")
        return None
    print("Test passed.")

    print("Started_evaluating {}...".format(nn_hamilton.__class__.__name__))
    _evaluation_batches = 50
    accuracy_scores = {}
    (accuracy_scores[EvaluationScores.ACCURACY_SCORE_TAGS.perc_hamilton_found],
        accuracy_scores[EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_cycles_found],
        accuracy_scores[EvaluationScores.ACCURACY_SCORE_TAGS.perc_full_walks_found],
        accuracy_scores[EvaluationScores.ACCURACY_SCORE_TAGS.perc_long_walks_found], _) \
        = EvaluationScores.compute_accuracy_scores(*EvaluationScores.evaluate_on_saved_data(nn_hamilton, _evaluation_batches, ["DATA"]))

    for index in range(len(sizes)):
        if sizes[index] in EVALUATION_SIZES:
            item_dict.update(
                {"{}_{}".format(score_name, sizes[index]): accuracy_scores[score_name][index] for score_name in
                 accuracy_scores})
    return item_dict
