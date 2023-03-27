import hamgnn.main_model as main_model

if __name__ == "__main__":
    model = main_model.load_main_model()
    print(f"Succes, you can find the model checkpoint at {main_model.MAIN_MODEL_CHECKPOINT_PATH}.")
