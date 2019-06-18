from flask import Flask, render_template, request
from wtforms import Form, StringField
# from keras.models import load_model
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)
app.config['SECRET_KEY'] = '0d38ae4150f11f348396cefafefe1294'

# tokenizer = Tokenizer()

class InputForm(Form):
    input_value = StringField('input_value')

    @app.route('/')
    @app.route('/<name>', methods=['GET', 'POST'])
    def func(name=None):

        switcher = {
            "books": "analyze.html",
            "electronics": "analyze.html",
            "kitchen": "analyze.html",
            "dvd": "analyze.html"
        }

        MAX_LEN_VALUE = {
            "books": 200,
            "electronics": 200,
            "kitchen": 200,
            "dvd": 200
        }

        # MAX_REVIEW_LENGTH_FOR_KERAS_RNN = MAX_LEN_VALUE.get(name, 0)

        form = InputForm(request.form)

        predictionResult = ""
        if request.method == 'POST':
            input_value = request.form['input_value']
            print(input_value)
            # processing ....

            # test_samples = input_value
            # tokenizer.fit_on_texts(test_samples)
            # test_samples_tokens = tokenizer.texts_to_sequences(test_samples)
            # test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN,
            #                                         padding='post')

            if name == "books":
                print(name)
                # result = gru_att_model_BOOKS.predict(x=test_samples_tokens_pad)
            elif name == "electronics":
                print(name)
                # result = gru_att_model_ELECTRONICS.predict(x=test_samples_tokens_pad)
            elif name == "kitchen":
                print(name)
                # result = gru_att_model_KITCHEN.predict(x=test_samples_tokens_pad)
            else:
                print(name)
                # result = gru_att_model_DVD.predict(x=test_samples_tokens_pad)

            predictionResult = "positive"
            # if result[0][0] > 0.5:
            #     predictionResult = "positive"
            # elif result[0][0] == 0.5:
            #     predictionResult = "neutral"
            # else:
            #     predictionResult = "negative"

        return render_template(switcher.get(name, "index.html"), name=name, form=form, res=predictionResult)

if __name__ == "__main__":
    app.run(debug=True)