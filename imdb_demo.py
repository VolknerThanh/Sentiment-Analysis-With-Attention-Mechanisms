from flask import Flask, render_template, request
from wtforms import Form, StringField
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras import backend as K
from keras.preprocessing import sequence


app = Flask(__name__)
app.config['SECRET_KEY'] = '0d38ae4150f11f348396cefafefe1294'

tokenizer = Tokenizer()

class InputForm(Form):
    input_value = StringField('input_value')

    @app.route('/')
    @app.route('/<name>', methods=['GET', 'POST'])

    def func(name=None):

        switcher = {
            "books": "analyze.html",
            "electronics": "analyze.html",
            "kitchen": "analyze.html",
            "dvd": "analyze.html",
            "imdb": "analyze.html"
        }

        MAX_LEN_VALUE = {
            "books": 200,
            "electronics": 200,
            "kitchen": 200,
            "dvd": 200,
            "imdb": 200
        }

        MAX_REVIEW_LENGTH_FOR_KERAS_RNN = MAX_LEN_VALUE.get(name, 0)

        form = InputForm(request.form)

        predictionResult = ""
        this_value=""
        if request.method == 'POST':
            input_value = request.form['input_value']
            this_value = input_value
            print(input_value)
            # processing ....



            if name == "books":
                print(name)
                # after_predict = gru_att_model_BOOKS.predict(x=test_samples_tokens_pad)
                # result = after_predict[0][0]
            elif name == "electronics":
                print(name)
                # after_predict = gru_att_model_ELECTRONICS.predict(x=test_samples_tokens_pad)
                # result = after_predict[0][0]
            elif name == "kitchen":
                print(name)
                # after_predict = gru_att_model_KITCHEN.predict(x=test_samples_tokens_pad)
                # result = after_predict[0][0]
            elif name == "kitchen":
                print(name)
                # after_predict = gru_att_model_DVD.predict(x=test_samples_tokens_pad)
                # result = after_predict[0][0]
            else:
                print(name)
                # result = model_predict("Models/imdb_model.h5", input_value, MAX_REVIEW_LENGTH_FOR_KERAS_RNN)

                saved_model = load_model("Models/imdb_model.h5")
                wordIndex = imdb.get_word_index()
                words = input_value.split()
                review = []
                for word in words:
                    if word not in wordIndex:
                        review.append(2)
                    else:
                        review.append(wordIndex[word] + 3)
                review = sequence.pad_sequences([review], maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)
                result = saved_model.predict(review)
                print("Prediction (0 = negative, 1 = positive) = ", end="")
                print("%0.4f" % result[0][0])
                del saved_model
                K.clear_session()

            # predictionResult = "positive"
            if result >= 0.5:
                predictionResult = "positive"
            else:
                predictionResult = "negative"

        return render_template(switcher.get(name, "index.html"), name=name, form=form, res=predictionResult, input_content=this_value)

if __name__ == "__main__":
    app.run(debug=True)