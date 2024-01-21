import streamlit as st
from PIL import Image
import plotly.graph_objects as go

st.write(
    """
# Stock Price Predictor Using LSTM:
Stock values is very valuable but extremely hard to predict correctly for any human being on their own. This project seeks to solve the problem of Stock Prices Prediction by utilizes Deep Learning models, Long-Short Term Memory (LSTM) Neural Network algorithm, to predict future stock values.\n
"""
)

st.sidebar.write(
    """
# Stock Price Prediction:
Predict Stock Price For The Next 30 Days
"""
)


def google():
    st.write("""# GOOGLE Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("GOOGL.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredictGOOGL.png")
    st.image(graph_image, width=500)


def apple():
    st.write("""# APPLE Stock Price Prediction""")

    st.write(
        """
# Dataset Sample:
Taken From yahoofinance\n
"""
    )

    import pandas as pd

    df = pd.read_csv("AAPL.csv")
    st.write(df)
    st.write("""# Dataset Analysis:""")
    graph_image = Image.open("newplot.png")
    st.image(graph_image)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredict.png")
    st.image(graph_image,width=550)


def hindustan():
    st.write("""# HINDUSTAN UNILEVER Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("HINDUNILVR.NS.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredicthindustan.png")
    st.image(graph_image, width=500)


def walmart():
    st.write("""# WALMART Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("WMT.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredictwalmart.png")
    st.image(graph_image, width=500)


def colgate():
    st.write("""# COLGATE-PALMOLIVE Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("CL.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredictCP.png")
    st.image(graph_image, width=500)


def infosys():
    st.write("""# INFOSYS Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("INFY.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredictinfy.png")
    st.image(graph_image, width=500)


def johnson():
    st.write("""# JOHNSON & JHONSON Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("JNJ.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredictj&j.png")
    st.image(graph_image, width=500)


def tatamotors():
    st.write("""# TATA MOTORS Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("TATAMOTORS.NS.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredicttata.png")
    st.image(graph_image, width=500)


def hdfc():
    st.write("""# HDFC BANK Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("HDB.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredictHDFC.png")
    st.image(graph_image, width=500)


def pg():
    st.write("""# P & G Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("PG.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredictp&g.png")
    st.image(graph_image, width=500)


def pepsi():
    st.write("""# PEPSICO Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("PEP.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredictpepsi.png")
    st.image(graph_image, width=500)


def disney():
    st.write("""# WALT DISNEYCO Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("DIS.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredictdis.png")
    st.image(graph_image, width=500)


def starbucks():
    st.write("""# STARBUCKS Stock Price Prediction""")

    st.write(
        """
    # Dataset Sample:
    Taken From yahoofinance\n
    """
    )

    import pandas as pd

    df = pd.read_csv("SBUX.csv")
    st.write(df)

    st.write(
        """
    # Prediction Graph:
    Predict Values For The Next 30 Days\n
    """
    )

    graph_image = Image.open("30daypredictsb.png")
    st.image(graph_image, width=500)


select_datasets = {
    "GOOGLE": google,
    "APPLE": apple,
    "HINDUSTAN UNILEVER": hindustan,
    "WALMART": walmart,
    "COLGATE-PALMOLIVE": colgate,
    "INFOSYS": infosys,
    "JOHNSON & JOHNSON": johnson,
    "TATA MOTORS": tatamotors,
    "PEPSI": pepsi,
    "P&G": pg,
    "HDFC BANK": hdfc,
    "WALT DISNEYCO": disney,
    "STARBUCKS": starbucks,
}

option = st.sidebar.selectbox("Select Dataset", (select_datasets.keys()))
select_datasets[option]()
