from flask import Flask, render_template, flash, request
from wtforms import Form, StringField, SubmitField
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import os
app = Flask(__name__)
app.config['SECRET_KEY'] = '0d38ae4150f11f348396cefafefe1294'

class InputForm(Form):
    input_value = StringField('input_value')

    @app.route('/')
    @app.route('/<name>', methods=['GET', 'POST'])

    def func(name=None):
        # switcher={
        #     "books": "books.html",
        #     "electronics": "electronics.html",
        #     "kitchen": "kitchen.html",
        #     "dvd": "dvd.html"
        # }

        # if name != "books" or name != "electronics" or name != "kitchen" or name != "dvd":
        #     return render_template("index.html", name=name)

        form = InputForm(request.form)
        tokenizer = Tokenizer()
        MAX_REVIEW_LENGTH_FOR_KERAS_RNN = 200

        predictionResult = ""
        if request.method == 'POST':
            input_value = request.form['input_value']
            print(input_value)

            # processing ....
            # predictionResult = input_value + "asssssss"
            loaded_model = load_model("Models/book_model.h5")
            test_samples = input_value
            tokenizer.fit_on_texts(test_samples)
            test_samples_tokens = tokenizer.texts_to_sequences(test_samples)
            test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN,
                                                    padding='post')

            # result = loaded_model.predict(x=test_samples_tokens_pad)
            # print(result[0][0])
            # predictionResult = str(result[0][0])

            if name == "books":
                print(name)
                result = loaded_model.predict(x=test_samples_tokens_pad)
            elif name == "electronics":
                print(name)
                # result = gru_att_model_ELECTRONICS.predict(x=test_samples_tokens_pad)
            elif name == "kitchen":
                print(name)
                # result = gru_att_model_KITCHEN.predict(x=test_samples_tokens_pad)
            else:
                print(name)
                # result = gru_att_model_DVD.predict(x=test_samples_tokens_pad)

            if result[0][0] >= 0.5:
                predictionResult = "positive"
            else:
                predictionResult = "negative"

        print(predictionResult)

        # return render_template(switcher.get(name, "index.html"), name=name, form=form, res=predictionResult)
        return render_template("analyze.html", name=name, form=form, res=predictionResult)

if __name__ == "__main__":
    app.run(debug=True)