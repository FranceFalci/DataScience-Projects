"""
MÓDULO DE INTERFACES (CONTRATOS)
--------------------------------
Este archivo actúa como los "Planos del Arquitecto" para el sistema de procesamiento.
Define las reglas obligatorias (Interfaces) que deben cumplir las clases concretas.

PRINCIPIOS DE DISEÑO:
1. Desacoplamiento: El sistema no depende de librerías específicas (ej. YOLO, OpenCV),
   sino de estas interfaces genéricas.
2. Polimorfismo: Podemos cambiar YOLO por ResNet o GradCAM por Lime sin romper el código,
   siempre que las nuevas clases respeten estos contratos.

COMPONENTES:
- IModelLoader: Define cómo debe comportarse cualquier cargador de modelos (Cargar y Predecir).
- IExplainer: Define cómo debe comportarse cualquier algoritmo de explicabilidad (Generar mapas de calor).
- IImageProcessor: Define las utilidades básicas de manipulación de imágenes (Base64 <-> Array).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class IModelLoader(ABC):
    """Interface to load models."""

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """return class, confidence and data visualization."""
        pass


class IExplainer(ABC):
    """Interface to explainability."""

    @abstractmethod
    def generate_heatmap(self, image: np.ndarray, model: Any, target_layer: Any) -> np.ndarray:
        pass


class IImageProcessor(ABC):
    """Interface to p¿image processing."""

    @abstractmethod
    def decode_base64(self, base64_string: str) -> np.ndarray:
        pass

    @abstractmethod
    def encode_base64(self, image: np.ndarray) -> str:
        pass
