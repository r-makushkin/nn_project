import matplotlib.pyplot as plt
import streamlit as st
# import imageio
from PIL import Image
import requests
import torch
import io
from models.sport_model import sport_model
from models.preprocessing import preprocess

@st.cache_resource()
def load_model():
    model = sport_model
    model.load_state_dict(torch.load('models/weights_sport.pth'))
    return model


LABELS_PATH = 'labels.txt'
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


def predict(img, labels):
    model.eval()
    img = preprocess(img)
    with torch.no_grad():
        pred = labels[model(img.unsqueeze(0)).softmax(dim=1).argmax()]
    return pred



model = load_model()



st.title('Дообучение модели ResNet для различных задач')

st.sidebar.header('Выберите страницу')
page = st.sidebar.selectbox("Выберите страницу", ['Команда', 'Классифкация "спортивных" изображений', "Классификация изображений клеток крови"])

# Основная часть приложения в зависимости от выбранной страницы
if page == "Команда":
    st.header('Команда ResNet')
    st.subheader('Роман - модель Resnet18 для классификации "спортивных" изображений')
    # roman = Image.open('pics/roman.jpg')
    # st.image(roman)
    st.subheader('Ерлан - модель Resnet для классификации изображений клеток крови')
    # erlan = Image.open('pics/erlan.jpg')
    # st.image(erlan)
    st.header('Исходная модель - ResNet')


elif page == 'Классифкация "спортивных" изображений':
    st.header('Классифкация "спортивных" изображений')
    st.subheader('Состав датасета')
    st.text('Тренировочный сет - 13493 изображений, разбитые на 100 категорий')
    st.text('Валидационный сет - 500 изображений, разбитые на 100 категорий')
    st.subheader('Примеры изображений')
    # image_1 = imageio.imread('pics/1.jpg')[:, :, :]
    # image_2 = imageio.imread('pics/2.jpg')[:, :, :]
    # image_3 = imageio.imread('pics/3.jpg')[:, :, :]
    image_1 = Image.open('pics/1.jpg')
    image_2 = Image.open('pics/2.jpg')
    image_3 = Image.open('pics/3.jpg')
    fig, axes = plt.subplots(1, 3, figsize=(8, 10))
    axes[0].imshow(image_1)
    axes[0].set_title("air hockey")
    axes[0].axis('off')
    axes[1].imshow(image_2)
    axes[1].set_title("billiards")
    axes[1].axis('off')
    axes[2].imshow(image_3)
    axes[2].set_title("wheelchair basketball")
    axes[2].axis('off')
    st.pyplot(fig)
    st.subheader('Что использовал?')
    st.text('Модель - ResNet18')
    st.text('Заменил и обучил только выходной fc-слой - nn.Linear(512, 100)')
    st.text('Оптимайзер - Adam c lr = 0.005')
    st.text('Количество эпох обучения - 20')
    # image_4 = imageio.imread('pics/sport_graph.png')
    image_4 = Image.open('pics/sport_graph.png')   
    st.image(image_4, caption='График обучения модели', use_column_width=True)
    st.subheader('Чего удалось добиться?')
    st.text('acc_train - 0.506, loss_test - 2.482')
    st.text('acc_valid - 0.748, loss_valid - 1.289')
    st.subheader('Веб-приложение')
    cur_image = st.file_uploader('Загрузите Ваше изображение')
    # cur_url = st.text_input("Введите URL изображения:")

    if cur_image:
        try:
            cur_image = Image.open(cur_image)
            cur_image_pred = predict(cur_image, labels)
            st.image(cur_image, caption=cur_image_pred, use_column_width=True)
        except Exception:
            st.error("Произошла ошибка при загрузке или классификации изображения. Попробуйте еще раз!")

    # if cur_url:
    #     try:
    #         response = requests.get(cur_url)
    #         image = Image.open(io.BytesIO(response.content))
    #         cur_image_pred = predict(image, labels)
    #         st.image(image, caption=cur_image_pred, use_column_width=True)
    #     except Exception:
    #         st.error("Произошла ошибка при загрузке или классификации изображения. Попробуйте еще раз!")
    

elif page == "Классификация изображений клеток крови":
    st.subheader("Классификация изображений клеток крови")
    

# streamlit run app.py