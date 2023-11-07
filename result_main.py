import result_learyt
import result_parsing
import result_logreg
import result_resub
import result_database
import result_output
import sqlite3

artikul = 30753769
# Подключаемся к базе данных
connection = sqlite3.connect('database.db')
cursor = connection.cursor()
def check_artikul(artikul):
    cursor.execute('SELECT * FROM products WHERE artikul=?', (artikul,))
    result = cursor.fetchone()
    if result:
        # Если элемент найден, возвращаем остальные поля
        return result[1:]  # Возвращаем все поля, кроме первого (id)
    else:
        return 0  # Если элемент не найден, возвращаем 0

#объединим все файлы
# отправляю артикул в функцию, чтобы она спарсила файл с комментариями
if check_artikul(artikul) == 0:
    file_first = 'some.csv'
    file_second = 'some_resubed.csv'
    result_parsing.f_parsing(artikul, file_first)
    result_resub.f_resub(file_first, file_second)
    result_learyt.f_learyt(file_second)
    list = result_logreg.f_logreg()

    total_count = list[0]
    count_positive = list[1]
    count_negative = list[2]

    result_database.f_database(artikul, total_count, count_positive, count_negative)
    #result_ranforest.f_ranforest()
else:
    db_info = check_artikul(artikul)
    total_count = db_info[0]
    count_positive = db_info[1]
    count_negative = db_info[2]
    result_output.f_output(total_count, count_positive, count_negative)