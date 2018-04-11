
"""
Título: SISTEMA PARA DETECÇÃO DO ESTADO DO OLHO
Aluno: Matheus Coimbra Moraes

Uso: python recognize.py --treinamento images/treinamento --teste images/teste
"""
import argparse

import cv2
import sys
from imutils import paths
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from localbinarypatterns import LocalBinaryPatterns

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--treinamento", required=True,
	help="path to the training images")
ap.add_argument("-e", "--teste", required=True,
	help="path to the testing images")
args = vars(ap.parse_args())


#inicializa o descritor LBP com as listas dos dados e das classes
desc = LocalBinaryPatterns(8, 4)
data = []
labels = []

# para cada imagem no conjunto de treinamento
i = 0
for imagePath in paths.list_images(args["treinamento"]):
    # carrega a imagem da pasta de treinamento,
    # converte em escala cinza
    image = cv2.imread(imagePath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = desc.describe(gray)


    #extrai a classe do caminho da imagem, no caso, olho aberto e fechado
    # e depois atualiza as listas

    labels.append(imagePath.split("\\")[-2])
    data.append(hist)


print("")
# definindo a taxa de split 30% teste e 70% treinamento (mais comum)
split_test_size = 0.30
# criando dados de treino e de teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(data, labels, test_size=split_test_size, random_state=42)

# Criando o modelo preditivo, no caso, o "Linear Support Vector Classification"
modelSVM = LinearSVC(C=1, random_state=50)
#Pré-processamento
scaler = StandardScaler()
X_treinoSVM = scaler.fit_transform(X_treino)
X_testeSVM = scaler.transform(X_teste)
modelSVM.fit(X_treinoSVM,Y_treino)
# verificando a exatidão no modelo nos dados de treino
#recebe os dados de treino e aplica o modelo para fazer as previsões
nb_predict_train = modelSVM.predict(X_treinoSVM)
print("Exatidão SVM (Treino): {0:.4f}".format(metrics.accuracy_score(Y_treino, nb_predict_train)))
print()
# verificando a exatidão no modelo nos dados de teste
nb_predict_teste = modelSVM.predict(X_testeSVM)
print("Exatidão SVM (Teste): {0:.4f}".format(metrics.accuracy_score(Y_teste, nb_predict_teste)))
print()
# criando uma Confusion Matrix
print("Confusion Matrix SVM")
print("{0}".format(metrics.confusion_matrix(Y_teste, nb_predict_teste, labels=["aberto", "fechado"])))
print("")
# relatório de classificação
print("Classification Report SVM")
print(metrics.classification_report(Y_teste, nb_predict_teste, labels=["aberto", "fechado"]))

# Criando o modelo preditivo, no caso, o naive bayes
modelG = GaussianNB()
# treinando o modelo
#fit-> construir e treinar o modelo
modelG.fit(X_treino, Y_treino)
# verificando a exatidão no modelo nos dados de treino
#recebe os dados de treino e aplica o modelo para fazer as previsões
nb_predict_train = modelG.predict(X_treino)
print("Exatidão naive bayes (Treino): {0:.4f}".format(metrics.accuracy_score(Y_treino, nb_predict_train)))
print()
# verificando a exatidão no modelo nos dados de teste
nb_predict_teste = modelG.predict(X_teste)
print("Exatidão naive bayes (Teste): {0:.4f}".format(metrics.accuracy_score(Y_teste, nb_predict_teste)))
print()
# criando uma Confusion Matrix
print("Confusion Matrix naive bayes")
print("{0}".format(metrics.confusion_matrix(Y_teste, nb_predict_teste, labels=["aberto", "fechado"])))
print("")
# relatório de classificação
print("Classification Report naive bayes")
print(metrics.classification_report(Y_teste, nb_predict_teste, labels=["aberto", "fechado"]))

#modelo com RandomForest
modelF = RandomForestClassifier(random_state=50, n_estimators=200, oob_score=True,n_jobs=-1, max_features="auto", min_samples_leaf=10)
modelF.fit(X_treino, Y_treino)
rf_predict_train = modelF.predict(X_treino)
print("Exatidão RandomForest (Treino): {0:.4f}".format(metrics.accuracy_score(Y_treino, rf_predict_train)))
print()
# verificando a exatidão no modelo nos dados de teste
rf_predict_teste = modelF.predict(X_teste)
print("Exatidão RandomForest (Teste): {0:.4f}".format(metrics.accuracy_score(Y_teste, rf_predict_teste)))
print()
# criando uma Confusion Matrix
print("Confusion Matrix RandomForest")
print("{0}".format(metrics.confusion_matrix(Y_teste, rf_predict_teste, labels=["aberto", "fechado"])))
print("")
# relatório de classificação
print("Classification Report RandomForest")
print(metrics.classification_report(Y_teste, rf_predict_teste, labels=["aberto", "fechado"]))


input("Enter para continuar...")
sys.exit(0)
# Classifica imagens no repositorio de testes
#para vizualização
i = 0
linear = 0
naiveBayes = 0
randomForest = 0
for imagePath in paths.list_images(args["teste"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    #classifica
    predictionG = modelG.predict([hist])[0]
    print("Naive bayes: %s"%predictionG)

    predictionF = modelF.predict([hist])[0]
    print("Random Forest: %s"%predictionF)


    predictionSVM = modelSVM.predict([hist])[0]
    print("SVM: %s" % predictionSVM)
    print("")
    #redimensiona a imagem apenas com finalidade de mostrar
    #o resultado do teste
    image = cv2.resize(image, (150, 150))
    text = "{}:{}\n{}:{}\n{}:{}".format(predictionG,"Naive Bayes",predictionF,"Rand Forest",predictionSVM,"SVM")
    y0, dy = 10, 14
    for i, linha in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(image,  linha, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1)
    cv2.imshow("Imagem olho", image)
    cv2.waitKey(0)
    i += 1

