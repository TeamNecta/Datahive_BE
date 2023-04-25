import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from flask_cors import CORS
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = "mysecretkey"

CORS(
    app,
    origins=[
        "http://localhost:3000",
        "https://datahive.rabil.me",
        "https://datahive.pages.dev",
    ],
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        df = pd.read_csv(file)

        # Replace '?' and '' with NaN
        df.replace("?", np.NaN, inplace=True)
        df.replace("", np.NaN, inplace=True)

        # Delete Rows that contain duplicates
        df.drop_duplicates(inplace=True)

        # Delete Columns that have single values
        for key, value in df.items():
            if len(df[key].unique()) == 1:
                del df[key]

        miss_data = df.isnull().sum()[df.isnull().sum() > 0]
        miss_data = miss_data.to_frame()
        miss_data.columns = ["No of Missing Values"]
        cols = list(miss_data.index)
        dataType = df.dtypes.apply(lambda x: x.name)
        json_string = json.dumps(dataType.to_dict())
        dfJSON = df.to_json()
        return jsonify(
            {
                "data": dfJSON,
                "dataType": json_string,
                "cols": cols,
                "columns": list(df.columns),
                "table": df.to_html(),
                "filename": file.filename,
                "datatype_table": df.dtypes.to_frame().to_html()
            }
        )

        # return render_template(
        #     "advance_cleaning.html",
        #     data=df,
        #     dataType=dataType.transpose(),
        #     cols=cols,
        #     columns=list(df.columns),
        # )
    return {}


@app.route("/advance_cleaning", methods=["GET", "POST"])
def advance_cleaning():
    if request.method == "POST":
        df_json = request.form["data"]
        df = pd.read_json(df_json)
        if request.form["action"] == "replace_missing":
            columns = request.form["replace_column"]
            method = request.form["replace_method"]
            for col in columns:
                if method == "mean":
                    avg = df[col].astype("float").mean(axis=0)
                    df[col].replace(np.NaN, avg, inplace=True)
                elif method == "freq":
                    freq = df[col].value_counts().idxmax()
                    df[col].replace(np.NaN, freq, inplace=True)
                elif method == "deleteRow":
                    df.dropna(subset=[columns[0]], axis=0, inplace=True)
                    df.reset_index(drop=True, inplace=True)

        elif request.form["action"] == "change_datatype":
            column = request.form.getlist("column")
            datatype = request.form["datatype"]
            for col in column:
                df[col] = df[col].astype(datatype)

        elif request.form["action"] == "normalize_data":
            cols = request.form.getlist("column")
            for col in cols:
                df[col] = df[col] / df[col].max()

        # elif request.form["action"] == "convert_categorical":
        #     columns = request.form.getlist("columns")
        #     dummyframe = pd.get_dummies(df, columns=columns, prefix=columns)
        #     # dummyframe2 = pd.get_dummies(df['aspiration'])
        #     # dummyframe2.rename(columns={'std':'aspiration-std','turbo':'aspiration-turbo'}, inplace=True)
        #     # merge the dummyframe2 to main data frame and remove th aspiration column
        #     df = pd.concat([df, dummyframe], axis=1)
        #     # df.drop('aspiration', axis=1, inplace=True)
        #     clean_message = "Categorical data converted to integer successfully!"

    miss_data = df.isnull().sum()[df.isnull().sum() > 0]
    miss_data = miss_data.to_frame()
    miss_data.columns = ["No of Missing Values"]
    cols = list(miss_data.index)
    dataType = df.dtypes.apply(lambda x: x.name).to_dict()
    json_string = json.dumps(dataType)
    dfJSON = df.to_json()
    return jsonify(
        {
            "data": dfJSON,
            "dataType": json_string,
            "cols": cols,
            "columns": list(df.columns),
        }
    )


@app.route("/visualization", methods=["GET", "POST"])
def visualization():
    return render_template("visualize.html")


@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    df_json = request.form["data"]
    df = pd.read_json(df_json)
    _dict = {}
    if request.method == "POST":
        if request.form["action"] == "check_correlation":
            col = request.form.get("target_column")
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            numeric_cols.remove(col)
            for keys in numeric_cols:
                pearson_coef, p_value = stats.pearsonr(df[keys], df[col])
                _dict[keys] = [pearson_coef, p_value]

        elif request.form["action"] == "SLR":
            lm = LinearRegression()
            colX = request.form.getlist("Column_X")
            colY = request.form.getlist("Column_Y")
            X = df[colX]
            Y = df[colY]
            lm.fit(X, Y)
            R2 = lm.score(X, Y)
            print(R2)
            Yhat = lm.predict(X)
            MSE = mean_squared_error(df[colX], Yhat)
            print(MSE)

            Yhat[0:5]
            width = 12
            height = 10
            plt.figure(figsize=(width, height))
            # sns.residplot(x=df[colX], y=df[colY], data=df)
            sns.regplot(x=colX[0], y=colY[0], data=df)
            plt.ylim(
                0,
            )
            plt.show()

        elif request.form["action"] == "MLR":
            lm = LinearRegression()
            colX = request.form.getlist("selected_columns[]")
            colY = request.form.getlist("Column_Y")
            X = df[colX]
            Y = df[colY]
            lm.fit(X, Y)
            R2 = lm.score(X, Y)
            print(R2)
            Yhat = lm.predict(X)
            MSE = mean_squared_error(df[colX], Yhat)
            print(MSE)

            Yhat[0:5]
            width = 12
            height = 10
            plt.figure(figsize=(width, height))
            # sns.residplot(x=df[colX], y=df[colY], data=df)
            sns.regplot(x=colX[0], y=colY[0], data=df)
            plt.ylim(
                0,
            )
            plt.show()

        # elif request.form["action"] == "Predict":
        #     col = request.form.get("target_column")
        #     model = LinearRegression()
        #     y_data = df[col]
        #     x_data = df.drop(col, axis=1 )
        #     x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

        #     print("number of test samples :", x_test.shape[0])
        #     print("number of training samples:",x_train.shape[0])

        #     model.fit(x_train, y_train)
        #     y_pred = model.predict(x_test)
            
        #     input_values = {}
        #     for column in x_data.columns:
        #         input_values[column] = request.form['column']

        #     input_values = input_values.to_frame()

        #     price = model.predict(input_values)[0]
        #     print("Predicted price for new instance:", price)

            


    dfJSON = df.to_json()
    return jsonify(
        {
            "data": dfJSON,
            "cols": list(df.columns),
            "dict": dict,
        }
    )

if __name__ == "__main__":
    app.run(debug=True)
