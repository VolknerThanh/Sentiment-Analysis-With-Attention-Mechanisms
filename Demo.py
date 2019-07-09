from flask import Flask, render_template, request
from wtforms import Form
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import backend as bkn


app = Flask(__name__)
app.config['SECRET_KEY'] = '0d38ae4150f11f348396cefafefe1294'


def saved_model(path, inpVal, MaxLen):
    model = load_model(path)
    wordIndex = imdb.get_word_index()
    words = inpVal.split()
    review = []
    for word in words:
        if word not in wordIndex:
            review.append(2)
        else:
            review.append(wordIndex[word] + 3)

    review = sequence.pad_sequences([review], maxlen=MaxLen)
    result = model.predict(review)
    print('Prediction (0 = negative, 1 = positive) = ', end=" ")
    print("%0.4f" % result[0][0])
    del model
    bkn.clear_session()
    return result

class InputForm(Form):



    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/imdb', methods=['GET', 'POST'])
    def analyzing(name='imdb'):

        Max_len_val = 200
        model_path = "Models/imdb_model.h5"
        form = InputForm(request.form)
        predictedResult = ''
        original = ""

        if request.method == 'POST':
            input_value = request.form['input_value']
            original = input_value

            print('Gia tri dau vao: ' + str(input_value))
            # res = InputForm()
            result = saved_model( model_path, input_value, Max_len_val)

            if result >= 0.5:
                predictedResult = "positive"
            else:
                predictedResult = "negative"

        return render_template('analyze.html', name=name, form=form, res=predictedResult, input_content=original )


if __name__ == "__main__":
    app.run(debug=True)