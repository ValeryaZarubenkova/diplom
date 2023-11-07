def f_output(total_count, positive_count, negative_count):

    # Вычисление процента положительных и отрицательных комментариев

    positive_percent = (positive_count / total_count) * 100
    negative_percent = (negative_count / total_count) * 100
    print(f'Процент положительных комментариев: {positive_percent:.2f}%')
    print(f'Процент отрицательных комментариев: {negative_percent:.2f}%')
    

    import nltk
    import csv
    forthemostpopular = []
    text = ''
    # # подключаем статистику 
    
    from nltk.probability import FreqDist
    with open('some_resubed.csv','r', encoding='utf-8') as file: 
        reader = csv.reader(file)
        for row in reader:
            for token in row[1].split():
                text = text + token.strip('\'[],') + ' '
    
    # # исчитаем слова в тексте по популярности
    words = nltk.tokenize.word_tokenize(text)
    fdist = FreqDist(words)
    print(fdist.most_common(10))
    # показываем самые популярные
    from wordcloud import WordCloud
    # и графический модуль, с помощью которого нарисуем это облако
    import matplotlib.pyplot as plt
    # переводим всё в текстовый формат
    text_raw = " ".join(words)
    # готовим размер картинки
    wordcloud = WordCloud(width=1600, height=800).generate(text_raw)
    plt.figure( figsize=(20,10), facecolor='k')
    # добавляем туда облако слов
    plt.imshow(wordcloud)
    # выключаем оси и подписи
    plt.axis("off")
    # убираем рамку вокруг
    plt.tight_layout(pad=0)
    # выводим картинку на экран
    plt.show()

    return [total_count, positive_count, negative_count]
