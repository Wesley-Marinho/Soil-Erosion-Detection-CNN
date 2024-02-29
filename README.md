Configuração dos modelos
batch_size = 4,
learning_rate = 2e-4,
epochs = 40

dropout = False
L2 Regularization = False
Early Stopping = False
Batch Normalization = False

data = semi augmented
------------------------------------------

DLinkNet34
Acurácia de validação: 0.9890994563935294
F1-score de validação: 0.793649301640202
IoU de validação: 0.9811525582532669
------------------------------------------

DLinkNet50
Acurácia de validação: 0.9845649711669437
F1-score de validação: 0.7056144686757015
IoU de validação: 0.9740713095995462
------------------------------------------

DLinkNet101
Acurácia de validação: 0.988018342426845
F1-score de validação: 0.7752098133256904
IoU de validação: 0.9795217952192948
------------------------------------------

Unet
Acurácia de validação: 0.9859630153292701
F1-score de validação: 0.7326487158499042
IoU de validação: 0.9762629229833124

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

Configuração dos modelos
batch_size = 4,
learning_rate = 2e-4,
epochs = 40

dropout = True
L2 Regularization = False
Early Stopping = False
Batch Normalization = False

data = full augmented
------------------------------------------

DLinkNet34

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

Configuração dos modelos
batch_size = 4,
learning_rate = 2e-4,
epochs = 40

Dropout = True
L2 Regularization = True
Early Stopping = True
Batch Normalization = True

data = full augmented
------------------------------------------

DLinkNet34