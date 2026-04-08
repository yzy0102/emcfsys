from emcfsys._training_tasks import SegmentationTrainingRequest, run_training_task


def test_run_training_task(monkeypatch):
    calls = {"update": [], "log": []}

    def fake_train_loop(images_dir, masks_dir, save_path, **kwargs):
        callback = kwargs["callback"]
        callback(1, 1, 3, 0.4)
        callback(
            1,
            0,
            3,
            0.3,
            finished_epoch=True,
            epoch_time=2.0,
            model_dict="model",
            metrics={"IoU": 0.9},
        )
        return save_path

    monkeypatch.setattr("emcfsys._training_tasks.train_loop", fake_train_loop)

    request = SegmentationTrainingRequest(
        images_dir="images",
        masks_dir="masks",
        save_path="save_dir",
        backbone_name="resnet34",
        model_name="deeplabv3plus",
        lr=1e-4,
        batch_size=2,
        epochs=5,
        device="cpu",
        classes_num=2,
        target_size=512,
        ignore_index=-1,
        pretrained_model=None,
    )

    logs = run_training_task(
        request,
        update_loss_curve=lambda loss, epoch=None: calls["update"].append((epoch, loss)),
        log=lambda message: calls["log"].append(message),
        stop_flag_fn=lambda: False,
    )

    assert len(logs) == 2
    assert calls["update"] == [(1, 0.3)]
    assert any("Estimated total training time" in message for message in calls["log"])
