import uvicorn
import webbrowser
import threading
import time

def start_browser():
    time.sleep(1)  # Sunucunun başlaması için bekle
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    threading.Thread(target=start_browser).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000)  # reload kaldırıldı