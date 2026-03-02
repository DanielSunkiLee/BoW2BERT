from transformers import Trainer, TrainingArguments


def train_model(model, small_train, small_eval, compute_metrics):
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        TENSORBOARD_LOGGING_DIR="./logs",
        logging_steps=100,
        eval_strategy="steps",
        save_strategy="steps",  
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        max_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train,
        eval_dataset=small_eval,
        compute_metrics = compute_metrics,
    )

    trainer.train()
    trainer.save_model("./saved_model")
    return trainer
