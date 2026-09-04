"""
Detector de Ruptura - Discontinuidad C¹ en Series Financieras
Detecta puntos de ruptura donde hay cambios abruptos en la derivada primera.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


class DetectorRuptura:
    """
    Detecta discontinuidades C¹ (rupturas de derivada) en series temporales.
    Utiliza análisis de segunda derivada para identificar cambios abruptos de pendiente.
    """
    
    def __init__(self, ventana_suave=5, threshold_ruptura=None):
        """
        Args:
            ventana_suave: tamaño de ventana para suavizado gaussiano
            threshold_ruptura: umbral para detectar ruptura (si None, usa percentil 95)
        """
        self.ventana_suave = ventana_suave
        self.threshold_ruptura = threshold_ruptura
    
    def calcular_derivada(self, serie, sigma=1.0):
        """
        Calcula primera derivada usando filtro gaussiano.
        Esto suaviza la serie antes de derivar para reducir ruido.
        
        Args:
            serie: array 1D de valores
            sigma: parámetro de suavizado gaussiano
        
        Returns:
            derivada primera
        """
        serie_suave = gaussian_filter1d(serie, sigma=sigma)
        derivada = np.gradient(serie_suave)
        return derivada
    
    def detectar_discontinuidades_c1(self, serie, sigma=1.0, metodo='segunda_derivada'):
        """
        Detecta puntos donde hay discontinuidad en la derivada primera (C¹).
        
        Args:
            serie: array 1D de valores
            sigma: suavizado gaussiano
            metodo: 'segunda_derivada' o 'cambio_pendiente'
        
        Returns:
            dict con índices de ruptura y magnitudes
        """
        derivada_1 = self.calcular_derivada(serie, sigma=sigma)
        
        if metodo == 'segunda_derivada':
            # Segunda derivada (cambio de derivada)
            derivada_2 = np.gradient(derivada_1)
            derivada_2_abs = np.abs(derivada_2)
            
        elif metodo == 'cambio_pendiente':
            # Cambio absoluto en pendiente local
            ventana = self.ventana_suave
            derivada_2_abs = np.abs(np.diff(derivada_1, n=1))
            derivada_2_abs = np.pad(derivada_2_abs, (1, 0), mode='edge')
        
        # Determinar threshold
        if self.threshold_ruptura is None:
            threshold = np.percentile(derivada_2_abs, 95)
        else:
            threshold = self.threshold_ruptura
        
        # Encontrar rupturas
        indices_ruptura = np.where(derivada_2_abs > threshold)[0]
        
        return {
            'indices': indices_ruptura,
            'magnitudes': derivada_2_abs[indices_ruptura],
            'derivada_1': derivada_1,
            'derivada_2': derivada_2_abs,
            'threshold': threshold
        }
    
    def analizar_ruptura(self, serie, indice_ruptura, ventana_contexto=10):
        """
        Analiza en detalle una ruptura específica.
        
        Args:
            serie: array 1D
            indice_ruptura: índice de la ruptura
            ventana_contexto: puntos antes/después para contexto
        
        Returns:
            dict con análisis de la ruptura
        """
        inicio = max(0, indice_ruptura - ventana_contexto)
        fin = min(len(serie), indice_ruptura + ventana_contexto + 1)
        
        ventana = serie[inicio:fin]
        der_1 = self.calcular_derivada(ventana)
        
        pendiente_antes = np.mean(der_1[:ventana_contexto])
        pendiente_despues = np.mean(der_1[ventana_contexto:])
        
        return {
            'indice': indice_ruptura,
            'valor': serie[indice_ruptura],
            'pendiente_antes': pendiente_antes,
            'pendiente_despues': pendiente_despues,
            'cambio_pendiente': pendiente_despues - pendiente_antes,
            'ventana': ventana
        }


# Ejemplo de uso
if __name__ == "__main__":
    # Crear serie de prueba con ruptura
    t = np.linspace(0, 10, 200)
    serie = np.sin(t) + 0.5 * t
    
    # Introducir una ruptura (cambio de pendiente abrupto)
    ruptura_idx = 100
    serie[ruptura_idx:] += 2 * (t[ruptura_idx:] - t[ruptura_idx])
    
    # Detectar
    detector = DetectorRuptura(ventana_suave=5)
    resultados = detector.detectar_discontinuidades_c1(serie, sigma=0.5)
    
    print(f"Rupturas detectadas en índices: {resultados['indices']}")
    print(f"Magnitudes: {resultados['magnitudes']}")
    print(f"Threshold usado: {resultados['threshold']:.4f}")
