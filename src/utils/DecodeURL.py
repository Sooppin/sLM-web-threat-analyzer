import os
import json

import urllib.parse


class DecodeURL:
    def __init__(self):
        pass

    @staticmethod
    def decode_file(input_file_path) -> str:
        output_file_path = os.path.splitext(input_file_path)[0] + "_decoded.done"
        seen_attack_syntax = set()

        required_fields = ['payload', 'attack_syntax', 'attack_type', 'dict_gpt_api_summary']

        with open(input_file_path, 'r', encoding='utf-8') as infile, \
                open(output_file_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    dict_data = json.loads(line.strip())
                    decoded_dict = dict_data.copy()

                    skip_record = False
                    for field in required_fields:
                        if field in decoded_dict and (decoded_dict[field] is None or decoded_dict[field] == ""):
                            skip_record = True
                            break
                    if skip_record:
                        continue

                    if 'payload' in decoded_dict:
                        decoded_dict['payload'] = urllib.parse.unquote(decoded_dict['payload'])

                    if 'attack_syntax' in decoded_dict:
                        decoded_dict['attack_syntax'] = urllib.parse.unquote(decoded_dict['attack_syntax'])

                        if decoded_dict['attack_syntax'] in seen_attack_syntax:
                            continue
                        seen_attack_syntax.add(decoded_dict['attack_syntax'])

                    outfile.write(json.dumps(decoded_dict, ensure_ascii=False) + '\n')
                except json.JSONDecodeError:
                    continue

        return output_file_path

    @staticmethod
    def decode_query(query: str) -> str:
        try:
            decoded_query = query

            decoded_query = urllib.parse.unquote(decoded_query)

            return decoded_query

        except Exception:
            return query
