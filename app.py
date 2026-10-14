
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from datetime import datetime, date
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# 1. CẤU HÌNH ỨNG DỤNG
# =========================
st.set_page_config(
    page_title="Dự đoán Cổ phiếu LSTM",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Dự đoán giá cổ phiếu bằng LSTM")
st.caption(
    "Dữ liệu lịch sử từ Yahoo Finance. "
    "Kết quả là dự báo thử nghiệm, không phải khuyến nghị đầu tư."
)

st.sidebar.header("Cấu hình dự đoán")

ticker = st.sidebar.text_input(
    "Mã cổ phiếu (AAPL, MSFT, VNM.VN...)",
    value="AAPL"
).strip().upper()

start_date = st.sidebar.date_input(
    "Ngày bắt đầu",
    value=date(2015, 1, 1)
)

end_date = st.sidebar.date_input(
    "Ngày kết thúc",
    value=datetime.today().date()
)

prediction_days = st.sidebar.slider(
    "Số ngày lịch sử dùng làm đầu vào",
    min_value=30,
    max_value=90,
    value=60
)

epochs = st.sidebar.slider(
    "Số vòng huấn luyện (epochs)",
    min_value=5,
    max_value=50,
    value=10
)


# =========================
# 2. TẢI DỮ LIỆU
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(stock_ticker, start, end):
    # Yahoo Finance thường không bao gồm ngày end.
    # Cộng 1 ngày để bao gồm ngày kết thúc người dùng chọn.
    download_end = (
        pd.Timestamp(end) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    return yf.download(
        stock_ticker,
        start=start.strftime("%Y-%m-%d"),
        end=download_end,
        auto_adjust=False,
        progress=False
    )


if not ticker:
    st.warning("Vui lòng nhập mã cổ phiếu.")
    st.stop()

if start_date >= end_date:
    st.error("Ngày bắt đầu phải trước ngày kết thúc.")
    st.stop()

try:
    with st.spinner("Đang tải dữ liệu từ Yahoo Finance..."):
        data = load_data(ticker, start_date, end_date)
except Exception as exc:
    st.error(f"Không thể tải dữ liệu: {exc}")
    st.stop()

if data.empty:
    st.error(
        "Không tìm thấy dữ liệu. Hãy kiểm tra mã cổ phiếu, "
        "khoảng thời gian hoặc kết nối mạng."
    )
    st.stop()


# =========================
# 3. CHUẨN HÓA DỮ LIỆU
# =========================
try:
    close_data = data["Close"]

    # Xử lý trường hợp yfinance trả về DataFrame
    # nhiều cấp cột thay vì một Series.
    if isinstance(close_data, pd.DataFrame):
        if ticker in close_data.columns:
            close_data = close_data[ticker]
        else:
            close_data = close_data.iloc[:, 0]

    df_close = close_data.to_frame(name="Close").dropna()
    df_close = df_close[
        np.isfinite(df_close["Close"])
    ]
except (KeyError, IndexError, ValueError) as exc:
    st.error(f"Không đọc được cột giá đóng cửa: {exc}")
    st.stop()

if len(df_close) <= prediction_days + 5:
    st.error(
        f"Chỉ có {len(df_close)} phiên dữ liệu hợp lệ. "
        f"Cần nhiều hơn {prediction_days + 5} phiên. "
        "Hãy chọn khoảng thời gian dài hơn hoặc giảm Window Size."
    )
    st.stop()

st.subheader(f"📊 Dữ liệu lịch sử: {ticker}")
st.write(f"Tổng số phiên hợp lệ: {len(df_close)}")
st.dataframe(df_close.tail(10), use_container_width=True)

history_fig = go.Figure()
history_fig.add_trace(
    go.Scatter(
        x=df_close.index,
        y=df_close["Close"],
        mode="lines",
        name="Giá đóng cửa"
    )
)
history_fig.update_layout(
    title=f"Lịch sử giá đóng cửa - {ticker}",
    xaxis_title="Ngày",
    yaxis_title="Giá",
    template="plotly_dark"
)
st.plotly_chart(history_fig, use_container_width=True)


# =========================
# 4. CHUẨN BỊ TRAIN / TEST
# =========================
dataset = df_close["Close"].to_numpy(dtype=np.float64).reshape(-1, 1)

training_data_len = int(len(dataset) * 0.8)

if training_data_len <= prediction_days:
    st.error("Tập huấn luyện không đủ dài cho Window Size đã chọn.")
    st.stop()

# Chỉ fit scaler trên tập train để tránh rò rỉ dữ liệu.
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(dataset[:training_data_len])

# Biến đổi toàn bộ dữ liệu bằng scaler đã fit trên train.
scaled_data = scaler.transform(dataset)

X_train, y_train = [], []

for i in range(prediction_days, training_data_len):
    X_train.append(
        scaled_data[i - prediction_days:i, 0]
    )
    y_train.append(scaled_data[i, 0])

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.float32)

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

# Tập test sử dụng các phiên ngay trước ranh giới train
# để tạo cửa sổ đầu vào cho phiên test đầu tiên.
test_data = scaled_data[
    training_data_len - prediction_days:
]

X_test = []

for i in range(prediction_days, len(test_data)):
    X_test.append(
        test_data[i - prediction_days:i, 0]
    )

X_test = np.asarray(X_test, dtype=np.float32)
X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

y_test = dataset[training_data_len:].reshape(-1, 1)


# =========================
# 5. HUẤN LUYỆN VÀ DỰ ĐOÁN
# =========================
st.subheader("🤖 Huấn luyện mô hình LSTM")

if st.button("Bắt đầu huấn luyện và dự đoán", type="primary"):

    try:
        with st.spinner("Đang huấn luyện mô hình..."):

            model = Sequential([
                LSTM(
                    50,
                    return_sequences=True,
                    input_shape=(prediction_days, 1)
                ),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dense(25, activation="relu"),
                Dense(1)
            ])

            model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            early_stopping = EarlyStopping(
                monitor="loss",
                patience=3,
                restore_best_weights=True
            )

            model.fit(
                X_train,
                y_train,
                batch_size=32,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=0,
                shuffle=False
            )

            # Dự đoán tập test
            scaled_predictions = model.predict(
                X_test,
                verbose=0
            )
            predictions = scaler.inverse_transform(
                scaled_predictions
            )

            # Dự đoán phiên kế tiếp
            last_days = scaled_data[-prediction_days:, 0]
            X_future = last_days.reshape(
                1, prediction_days, 1
            )

            future_scaled = model.predict(
                X_future,
                verbose=0
            )
            future_price = scaler.inverse_transform(
                future_scaled
            )[0, 0]

        # =========================
        # 6. ĐÁNH GIÁ MÔ HÌNH
        # =========================
        mae = mean_absolute_error(
            y_test.ravel(),
            predictions.ravel()
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test.ravel(),
                predictions.ravel()
            )
        )

        latest_close = float(dataset[-1, 0])

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE trên tập test",
            f"{mae:,.2f}"
        )
        col2.metric(
            "RMSE trên tập test",
            f"{rmse:,.2f}"
        )
        col3.metric(
            "Giá đóng cửa gần nhất",
            f"{latest_close:,.2f}"
        )

        st.metric(
            "Giá dự đoán cho phiên tiếp theo",
            f"{future_price:,.2f}",
            delta=f"{future_price - latest_close:,.2f}"
        )

        # =========================
        # 7. VẼ BIỂU ĐỒ TEST
        # =========================
        train = df_close.iloc[:training_data_len]
        valid = df_close.iloc[training_data_len:].copy()

        valid["Predictions"] = predictions.ravel()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=train.index,
            y=train["Close"],
            name="Dữ liệu train",
            mode="lines"
        ))

        fig.add_trace(go.Scatter(
            x=valid.index,
            y=valid["Close"],
            name="Giá thực tế (test)",
            mode="lines"
        ))

        fig.add_trace(go.Scatter(
            x=valid.index,
            y=valid["Predictions"],
            name="Giá dự đoán",
            mode="lines"
        ))

        fig.update_layout(
            title=f"Kết quả dự đoán giá cổ phiếu {ticker}",
            xaxis_title="Ngày",
            yaxis_title="Giá đóng cửa",
            legend_title="Chú thích",
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Bảng so sánh thực tế và dự đoán
        st.subheader("📋 So sánh giá thực tế và dự đoán")
        st.dataframe(
            valid.tail(30),
            use_container_width=True
        )

        st.caption(
            "Lưu ý: MAE và RMSE được tính theo đơn vị giá của mã cổ phiếu. "
            "Dự báo một phiên tiếp theo chỉ là ước lượng từ mô hình, "
            "không đảm bảo phản ánh giá thực tế."
        )

    except Exception as exc:
        st.error(f"Đã xảy ra lỗi trong quá trình huấn luyện: {exc}")
