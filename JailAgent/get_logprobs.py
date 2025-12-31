import http.client
import json


def call_openai_api(prompt,
                    model="gpt-3.5-turbo-instruct",
                    max_tokens=50,
                    temperature=0,
                    top_p=0.9,
                    n=1,
                    logprobs=3,
                    stream=False):

   conn = http.client.HTTPSConnection("XXX") # Replace with your conn

   payload = json.dumps({
      "model": model,
      "prompt": prompt,
      "max_tokens": max_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "n": n,
      "stream": stream,
      "logprobs": logprobs
   })

   headers = {
      'Authorization': 'XXX',  # Replace with your key
      'Content-Type': 'application/json'
   }

   conn.request("POST", "/v1/completions", payload, headers)
   res = conn.getresponse()
   data = res.read()
   conn.close()

   try:
      return json.loads(data.decode("utf-8"))
   except json.JSONDecodeError:
      return {"error": "Invalid response", "raw": data.decode("utf-8")}


if __name__ == "__main__":
   response = call_openai_api("Who are you?", max_tokens=10)
   print(json.dumps(response, ensure_ascii=False, indent=2))
