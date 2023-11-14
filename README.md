# diplom
ВКРБ Зарубенковой Валерии "Программная система определения оценки настроения комментариев" с использованием алгоритмов машинного обучения. 
Для обучения модели машинного обучения не были использованы готовые датасеты, для лучшей работы системы было принято решение собрать реальные отзывы с крупнейших маркетплейсов и подготовить собственный датасет.
Для сбора комментариев должны быть разработаны собственные алгоритмы парсинга данных для разных сайтов, так как они имеют уникальную html - структуру и текст отзывов и их оценки расположены под разными тэгами. 
В изначальном датасете было собрано более 100 тыс. комментариев, из которых 75-80% содержали положительные отзывы. При таком соотношении модель отдавала предпочтение большей группе данных и чаще выдавала ложноположительные результаты. 
По этой причине с помощью добавления только негативных отзывов была проведена балансировка датасета, в результате которой количество положительных отзывов стало 55-60%.
Для того, чтобы при обучении модели ей не мешал «шум» данных, была выполнена обработка собранных отзывов, в которую вошло удаление цифр, английских букв и символов, приведение текста к единому регистру и его лемматизация, удаление стоп-слов, а также токенизация слов.
На вход модели машинного обучения нельзя подать текст, поэтому его необходимо представить в виде векторов. Для векторизации текста был использован Word2Vec.
На разных данных алгоритмы машинного обучения с учителем проявляют себя по-разному, поэтому важно выбрать наиболее подходящий алгоритм для конкретных данных. Это можно сделать с помощью метрик оценки качества алгоритмов. 
Для получения наилучшей модели было решено  алгоритмы: логистическая регрессия (Logistic Regression), метод опорных векторов (Support Vector Machine), случайный лес (Random Forest), метод k-ближайших соседей (k-Nearest Neighbors) и градиентный бустинг (Gradient Boosting Classifier).


Примечание: текст описания репозитория взят из моей научной статьи.

