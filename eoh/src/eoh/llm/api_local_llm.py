# This file includes classe to get response from deployed local LLM
import json
from typing import Collection
import requests


class InterfaceLocalLLM:
    """Language model that predicts continuation of provided source code.
    """

    def __init__(self, url, model_LLM=None):
        self._url = url  # 'http://127.0.0.1:11045/completions'
        if model_LLM:
            self.model_LLM = model_LLM
        else:
            self.model_LLM = "/home/aims/Workspace4/Llama-3.3-70B-Instruct"


    def get_response(self, content: str) -> str:
        while True:
            try:
                if 'chat' in self._url:
                    response = self._do_vllm_request(content)
                else:
                    response = self._do_request(content)
                return response
            except Exception as e:
                print(e)
                continue

    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference (inorder to decease the sample delay)
        data = {
            # 'model': self.model_LLM,
            'prompt': content,
            'repeat_prompt': 1,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        print(response)
        if response.status_code == 200:
            response = response.json()['content'][0]
            return response
        
    def _do_vllm_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference (inorder to decease the sample delay)
        data={
		"model": self.model_LLM,
		"messages": [
			{
				"role": "user",
				"content": content
			}
		    ]
	    }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        print(f'======================================== Response Content ========================================')
        print(f'{response}\n')
        print(f'==================================================================================================')

        if response.status_code == 200:
            response = response.json()['choices'][0]['message']['content']
            return response
