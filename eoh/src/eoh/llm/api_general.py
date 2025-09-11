import http.client
import json
import traceback

class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content):
        if 'gpt' in self.model_LLM:
            payload_explanation = json.dumps(
                {
                    "model": self.model_LLM,
                    "messages": prompt_content if isinstance(prompt_content, list) else [{"role": "user", "content": prompt_content}],
                }
            )

            headers = {
                "Authorization": "Bearer " + self.api_key,
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                "Content-Type": "application/json",
                "x-api2d-no-cache": 1,
            }

        else:
            payload_explanation = json.dumps(
                {
                "contents": [{
                    "parts":[{"text": prompt_content}]
                              }]
                }
            )

            headers = {
                # "key": self.api_key,
                # "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                "Content-Type": "application/json",
                # "x-api2d-no-cache": 1,
            }
            
        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                conn = http.client.HTTPSConnection(self.api_endpoint)

                if 'gpt' in self.model_LLM:
                    conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                else:
                    endpoint = f"/v1beta/models/{self.model_LLM}:generateContent?key={self.api_key}"
                    conn.request("POST", endpoint, payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                if 'gpt' in self.model_LLM:
                    response = json_data["choices"][0]["message"]["content"]
                else: # for Gemini
                    response = json_data["candidates"][0]["content"]["parts"][0]["text"]
                break
            except Exception as e:
                if self.debug_mode:
                    print("Error in API. Restarting the process...")
                    print(f"{traceback.format_exc()}")
                continue
            

        return response