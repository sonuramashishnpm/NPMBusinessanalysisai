from flask import Flask,render_template

app=Flask(__name__)

@app.route("/")
def web():
  return render_template("index.html")

if __name__=="main":
  port=int(os.environ.get("PORT",5000))
  app.run(host="0.0.0.0", port=port, debug=False)
