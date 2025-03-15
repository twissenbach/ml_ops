import mlflow

from model_serving.models.prediction import Model


class MLFlowGateway:

    models = {}

    def init_app(self, app):
        mlflow.set_tracking_uri(app.config['TRACKING_URI'] + '/mlruns')
        self.models = app.config['MODELS']

        for model in self.models.keys():
            for version in self.models[model].keys():
                flavor_ = self.models[model][version].get('mlflow_flavor', 'pyfunc')
                self.models[model][version]["model"] = self._load_model(self._get_model_uri(model, version), flavor_)

                if self.models[model][version].get('explainer', False):
                    from mlflow.client import MlflowClient

                    run_id = MlflowClient().get_model_version(model, version).run_id
                    self.models[model][version]['explainer'] = self._load_model(self._get_explainer_uri(run_id))

    def _get_model_uri(self, model_name, model_version):
        return f"models:/{model_name}/{model_version}"
    
    def _get_explainer_uri(self, run_id):
        return f"runs:/{run_id}/model/explainer"

    def _load_model(self, model_uri, model_flavor):
        if model_flavor == "sklearn":
            return mlflow.sklearn.load_model(model_uri)
        elif model_flavor == "tensorflow":
            return mlflow.tensorflow.load_model(model_uri)
        elif model_flavor == "pytorch":
            return mlflow.pytorch.load_model(model_uri)
        elif model_flavor == "xgboost":
            return mlflow.xgboost.load_model(model_uri)
        elif model_flavor == "pyfunc":
            return mlflow.pyfunc.load_model(model_uri)
        else:
            raise ValueError(f"Unsupported model flavor: {model_flavor}")

    def get_model(self, model_name, model_version) -> Model:

        model_ = self.models[model_name][model_version]

        model = Model(
            model_type=model_["model_type"]
            , model_name=model_name
            , model_version=model_version
            , threshold=model_["threshold"]
        )

        model.model = model_['model']
        model.labels = model_["labels"]

        return model


mlflow_gateway: MLFlowGateway = MLFlowGateway()
