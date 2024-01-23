
from typing import Any, Dict, Sequence, Tuple, Union, cast

import data
import torch

from torch import nn

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext

import torch.optim as optim
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from deepregressionmodel import DeepRegressionModel
from CustomDataset import CustomDataset
from azure.storage.blob import BlobServiceClient

import joblib
import pytz



TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class DispRecursos(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        useElastic=False
        self.context = context
        combined_df=joblib.load('combineddf_determined_20240122.pkl')
        print(str(combined_df.columns))
        # Codificar las variables categóricas usando one-hot encoding
        encoder = OneHotEncoder(sparse=False)
        categorical_columns = ["Month","Weekday","Hour"]  # Cambia esto a tus columnas categóricas
        combined_df2=encoder.fit_transform(combined_df[categorical_columns])
        

        joblib.dump(encoder, '/tmp/encoder.pkl')
        # Sube un archivo al contenedor
        combined_df=combined_df[["AppName"
                                 ,"Energy"
                                 ,"UsedMemory"
                                 ,"Energy-1","Energy-2","Energy-3"
                                 ,"Energyfut"
                                 ,"Mempre-1","Mempre-2","Mempre-3"
                                ,"Memfut","UserName"]].join(pd.DataFrame(combined_df2))
        print(str(combined_df))
        #combined_df = pd.get_dummies(combined_df, columns=["Weekday", "Month"], drop_first=True)
        combined_df = combined_df.iloc[3:-3]
        
        X = combined_df.drop(["Energyfut", "Memfut"], axis=1).values
        y_cpu = combined_df["Energyfut"].values
        y_mem = combined_df["Memfut"].values
        self.X_train, self.X_test, self.y_cpu_train, self.y_cpu_test,self.y_mem_train, self.y_mem_test = train_test_split(
            X, y_cpu, y_mem, test_size=0.2, random_state=42
        )

        # Crear una instancia del modelo
        input_dim = len(self.X_train[0])  # Ajusta según el número de características en tus datos
        self.model = self.context.wrap_model(DeepRegressionModel(input_dim))
        self.optimizer=self.context.wrap_optimizer(optim.Adam(self.model.parameters(), lr=0.001))

    def build_training_data_loader(self) -> DataLoader:

        # Normalizar los datos
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)

        # Convertir los datos a tensores de PyTorch
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        
        y_cpu_train_tensor = torch.tensor(self.y_cpu_train, dtype=torch.float32).view(-1, 1)
        y_mem_train_tensor = torch.tensor(self.y_mem_train, dtype=torch.float32).view(-1, 1)
        #Xtrain=(X_train_tensor,y_cpu_train_tensor)

        Xtrain = CustomDataset(X_train_tensor, y_cpu_train_tensor)
        
        return DataLoader(Xtrain, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> DataLoader:
        # Normalizar los datos
        
        X_test_scaled = self.scaler.transform(self.X_test)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_cpu_test_tensor = torch.tensor(self.y_cpu_test, dtype=torch.float32).view(-1, 1)
        y_mem_test_tensor = torch.tensor(self.y_mem_test, dtype=torch.float32).view(-1, 1)
        #Xtest=(X_test_tensor,y_cpu_test_tensor)
        Xtest = CustomDataset(X_test_tensor, y_cpu_test_tensor)
        return DataLoader(Xtest, batch_size=self.context.get_per_slot_batch_size())

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch
        # Definir función de pérdida y optimizador
        criterion = nn.MSELoss()  # Error cuadrático medio como función de pérdida
        #optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Optimizador Adam
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(data)  # X_train_tensor son tus datos de entrenamiento
        loss = criterion(outputs, labels)  # Calcular la pérdida
        print(str(loss))
        loss.backward()  # Retropropagación
        self.optimizer.step()  # Actualizar los parámetros
        torch.save(self.model.state_dict(), '/tmp/model.pth')
        output = self.model(data)
        loss = torch.nn.functional.mse_loss(output, labels)
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)
        return {"loss": loss}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = self.model(data)
        print(str(output))
        print(str(labels))
        validation_loss = torch.nn.functional.mse_loss(output, labels).item()

        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(data)

        return {"validation_loss": validation_loss, "accuracy": accuracy}
