from pipeline import Pipeline

if __name__=="__main__":
    data_path = "/data"
    pipeline = Pipeline(
        input_dir=data_path,
        output_dir=data_path
    )
    pipeline.train()
