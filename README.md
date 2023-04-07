# Estudo de caso: previsão de vendas

## Instruções para inferência

Este repositório contém os scripts preprocess.py, train.py e inference.py e o arquivo configs.py com configurações pré especificadas.
Para retreinar o modelo, atualize no arquivo configs.py o caminho para o conjunto de dados, que deve estar no formato csv. Depois, execute o script preprocess.py para fazer o pré-processamento dos dados. Se bem sucedido, execute o script train.py. Esta execução deve retornar um dicionário com as métricas de desempenho, o modelo serializado no formato .pkl e a lista de features utilizada no treino.

Para realizar o forecasting de novos dados, eles devem ser fornecidos no formato de um dataframe do pandas. Faça o import da função de inferência e passe o dataframe como parâmetro, do seguinte modo:

```python

from inference import predict_xgb

predict_xgb(new_data)

```

As predições serão salvas dentro da pasta "data" no fomato xlsx.