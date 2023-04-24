import cv2
import numpy as np
import telegram
import time

# Configuração do bot do Telegram
bot = telegram.Bot('SEU_TOKEN_DO_TELEGRAM')
chat_id = 'SEU_CHAT_ID'

# Carregando o modelo YOLO pré-treinado
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Definindo as classes que o modelo é capaz de detectar
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Definindo as cores para cada classe
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Configurações da webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Loop principal
while True:
    # Capturando uma imagem da webcam
    ret, img = cap.read()
    if img is None:
        continue

    # Obtendo as dimensões da imagem
    height, width, channels = img.shape

    # Convertendo a imagem para um blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)

    # Passando o blob pelo modelo YOLO
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Inicializando as listas de detecções
    boxes = []
    confidences = []
    class_ids = []

    # Percorrendo as detecções
    for out in outs:
        for detection in out:
            # Obtendo as probabilidades de cada classe
            scores = detection[5:]
            # Identificando a classe com a maior probabilidade
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Ignorando detecções fracas
            if confidence > 0.5 and class_id == 0:
                # Calculando as coordenadas da caixa delimitadora
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Adicionando as detecções à lista
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicando a supressão não-máxima para remover detecções redundantes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Desenhando as caixas delimitadoras e etiquetas nas detecções
    for i in range(len(boxes)):
        # Verificando se há detecções de pessoas
        if len(indexes) > 0:
            # Selecionando apenas as detecções de pessoas
            person_boxes = [boxes[i] for i in indexes.flatten() if class_ids[i] == 0]
            # Desenhando as caixas delimitadoras e etiquetas nas detecções de pessoas
            for i, box in enumerate(person_boxes):
                x, y, w, h = box
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                label = f"Person {i + 1}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Salvando a foto da pessoa detectada
                person_img = img[y:y + h, x:x + w]
                cv2.imwrite("person.jpg", person_img)
                # Enviando a foto da pessoa para o chat do Telegram
                bot.send_photo(chat_id=chat_id, photo=open('person.jpg', 'rb'))
                # Aguardando um intervalo de tempo para não enviar fotos repetidamente
                time.sleep(60)

        # Mostrando a imagem na tela
        cv2.imshow("Camera", img)

        # Verificando se o usuário pressionou a tecla 'q' para sair do programa
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
