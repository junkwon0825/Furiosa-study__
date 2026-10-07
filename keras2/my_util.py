import csv
import datetime
import os
import sys


def record_model_csv(
    model,
    data_shape,
    random_num,
    batch_size,
    history,
    training_time,
    test_loss=None,
    r2=None,
    rmse=None,
    test_acc=None,
    csv_file_path="model_history_log_v2.csv"
):

    # =====================================================
    # 1. 현재 날짜 / 시간
    # =====================================================
    current_time = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # =====================================================
    # 2. 실행 파일명
    # =====================================================
    file_name = os.path.basename(sys.argv[0])


    # =====================================================
    # 3. 모델 이름
    # =====================================================
    model_name = model.name


    # =====================================================
    # 4. 모델 구조
    # =====================================================
    structure_parts = []

    prev_layer_type = None

    for layer in model.layers:

        layer_type = layer.__class__.__name__

        # Dense
        if hasattr(layer, "units"):
            dim_str = str(layer.units)

        # Conv 계열
        elif hasattr(layer, "filters"):
            dim_str = str(layer.filters)

        else:
            dim_str = "?"

        # 같은 Layer가 연속이면 숫자만 기록
        if layer_type == prev_layer_type:
            structure_parts.append(dim_str)

        else:
            structure_parts.append(
                f"{layer_type} {dim_str}"
            )

            prev_layer_type = layer_type


    model_structure = " -> ".join(
        structure_parts
    )


    # =====================================================
    # 5. Loss / Optimizer
    # =====================================================
    if isinstance(model.loss, str):

        loss_func = model.loss

    else:

        loss_func = getattr(
            model.loss,
            "__name__",
            str(model.loss)
        )


    optimizer_name = getattr(
        model.optimizer,
        "name",
        type(model.optimizer).__name__
    )


    # =====================================================
    # 6. History 가져오기
    # =====================================================
    hist = history.history

    loss_list = hist.get(
        "loss",
        []
    )

    val_loss_list = hist.get(
        "val_loss",
        []
    )


    # accuracy / acc 둘 다 대응
    acc_list = hist.get(
        "accuracy",
        hist.get("acc", [])
    )

    val_acc_list = hist.get(
        "val_accuracy",
        hist.get("val_acc", [])
    )


    # =====================================================
    # 7. Epoch
    # =====================================================
    epochs = len(loss_list)


    # =====================================================
    # 8. Train Loss
    # =====================================================
    first_loss = (
        loss_list[0]
        if loss_list
        else None
    )

    last_loss = (
        loss_list[-1]
        if loss_list
        else None
    )


    # =====================================================
    # 9. Validation Loss
    # =====================================================
    if val_loss_list:

        best_val_loss = min(
            val_loss_list
        )

        best_val_epoch = (
            val_loss_list.index(
                best_val_loss
            )
            + 1
        )

        last_val_loss = (
            val_loss_list[-1]
        )

    else:

        best_val_loss = None
        best_val_epoch = None
        last_val_loss = None


    # =====================================================
    # 10. Accuracy
    # =====================================================
    last_accuracy = (
        acc_list[-1]
        if acc_list
        else None
    )


    if val_acc_list:

        best_val_accuracy = max(
            val_acc_list
        )

        best_val_acc_epoch = (
            val_acc_list.index(
                best_val_accuracy
            )
            + 1
        )

    else:

        best_val_accuracy = None
        best_val_acc_epoch = None


    # =====================================================
    # 11. evaluate() 결과 처리
    # =====================================================

    # return_dict=True 사용했을 경우
    if isinstance(test_loss, dict):

        result_dict = test_loss

        real_test_loss = result_dict.get(
            "loss"
        )

        # accuracy 또는 acc
        if test_acc is None:

            test_acc = result_dict.get(
                "accuracy",
                result_dict.get(
                    "acc"
                )
            )

    else:

        real_test_loss = test_loss


    # =====================================================
    # 12. CSV 기록 데이터
    # =====================================================
    log_data = [

        current_time,

        file_name,

        str(data_shape),

        random_num,

        model_name,

        model_structure,

        loss_func,

        optimizer_name,

        epochs,

        batch_size,

        round(
            training_time,
            4
        ),

        first_loss,

        last_loss,

        last_val_loss,

        best_val_loss,

        best_val_epoch,

        last_accuracy,

        best_val_accuracy,

        best_val_acc_epoch,

        real_test_loss,

        test_acc,

        rmse,

        r2
    ]


    # =====================================================
    # 13. CSV 저장
    # =====================================================
    file_exists = os.path.isfile(
        csv_file_path
    )


    with open(
        csv_file_path,
        mode="a",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)


        # 파일 없을 때 Header 생성
        if not file_exists:

            writer.writerow([

                "time",

                "pyfile",

                "data_shape",

                "random_state",

                "model",

                "model_structure",

                "loss",

                "optimizer",

                "epochs",

                "batch_size",

                "train_second",

                "first_loss",

                "last_loss",

                "last_val_loss",

                "best_val_loss",

                "best_val_epoch",

                "last_accuracy",

                "best_val_accuracy",

                "best_val_acc_epoch",

                "test_loss",

                "test_accuracy",

                "rmse",

                "r2"
            ])


        writer.writerow(
            log_data
        )


    print(
        f"모델 학습 정보가 "
        f"'{csv_file_path}'에 기록되었습니다."
    )