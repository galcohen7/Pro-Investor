# Add Python 3.13 to the current session PATH (no restart needed)
$env:PATH = "C:\Users\ASUS\AppData\Local\Programs\Python\Python313\Scripts;" +
            "C:\Users\ASUS\AppData\Local\Programs\Python\Python313;" +
            $env:PATH

streamlit run app.py
