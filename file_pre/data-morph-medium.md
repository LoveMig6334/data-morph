# Data Morph: Open Source File Data Migration with a Fine-tuned Small Language Model

*AI Builders 2026 · Track: Agentic AI + NLP*

[ภาพปก: cover image — โลโก้/กราฟิก Data Morph]

Data Morph คือเครื่องมือที่ถูกพัฒนาให้ developer สามารถแปลงรูปแบบไฟล์ข้อมูลระหว่างไฟล์ CSV, JSON และ TXT ที่มีรูปแบบของข้อมูลไม่สม่ำเสมอได้บนเครื่องตัวเองแบบ local ฟรี โดยไม่จำเป็นต้องพึ่งพา LLM หรือเครื่องมือ Agentic CLI ที่มีต้นทุนสูงทุกครั้งที่เรียกใช้งาน โดยการนำเทคนิค Knowledge Distillation มากลั่นความสามารถของ model อย่าง Claude Opus 4.8 ลงสู่ Small Language Model อย่าง Gemma 4 2B แล้วนำไป fine-tune ให้ทำงานแบบเดียวกันได้บนคอมพิวเตอร์ของผู้ใช้ บทความนี้จะเล่าเส้นทางการพัฒนาของ project ตั้งแต่การตั้งโจทย์ การวาง baseline การออกแบบ pipeline การ fine-tune model จนถึงการย่อโมเดลเพื่อนำไป deploy จริง

---

## Abstract

ในงานของ developer, data scientist และนักวิจัย AI หลาย ๆ คน การรวบรวมและจัดการทำความสะอาดข้อมูลจำนวนมากเป็นหนึ่งในขั้นตอนการทำงานที่มีความสำคัญมากและหลีกเลี่ยงไม่ได้ เพราะข้อมูลที่เราไปรวบรวมมาส่วนใหญ่จะกระจัดกระจายกันอยู่ในหลาย ๆ ที่ ซึ่งทำให้ไฟล์ส่วนมากยังไม่สามารถนำมาใช้งานได้ทันที และจำเป็นต้องมีการแปลงรูปแบบข้อมูลของไฟล์ โดยเฉพาะระหว่าง CSV, JSON และ TXT ซึ่งเป็น format ที่นิยมใช้เก็บข้อมูลกันมากที่สุด โดยในอดีตวิธีการที่เราใช้ในการทำความสะอาดข้อมูลให้เป็นมาตรฐานเดียวกันต้องอาศัยการเขียน script แปลงไฟล์เองด้วยมือในทุกกรณี และต่อมาในปัจจุบันเริ่มมีการนำเครื่องมือ เช่น Agentic CLI อย่าง Claude Code หรือ Codex มาช่วยอ่านไฟล์ เขียน script และตรวจผลลัพธ์ให้โดยอัตโนมัติ อย่างไรก็ตาม แนวทางแรกนั้นเสียเวลาและไม่ยืดหยุ่นกับข้อมูลที่มีโครงสร้างไม่สม่ำเสมอ ทำให้อาจเกิด error ที่ต้องเขียน script รองรับในหลาย ๆ รูปแบบ ในขณะที่แนวทางที่สองแม้จะจัดการงานที่ไม่สม่ำเสมอได้ดี แต่กลับมีต้นทุนค่า API ที่สูง และข้อจำกัดของ subscription เมื่อต้องใช้งานกับข้อมูลที่มีขนาดใหญ่ อีกทั้งข้อมูลที่ AI อ่านในขณะทำงานอาจมีเนื้อหาที่ sensitive หรืออาจมีข้อมูลส่วนตัวของผู้ใช้ปนอยู่ ซึ่งบริษัทเจ้าของ Model AI อาจเก็บข้อมูลไปใช้ในการ train model รุ่นต่อ ๆ ไป ซึ่งเป็นการละเมิดความเป็นส่วนตัวโดยที่ผู้ใช้อาจไม่รู้ตัวด้วยซ้ำ ช่องว่างนี้คือเหตุผลที่โครงงานนี้ต้องการพัฒนาเครื่องมือที่สามารถทำงานเดียวกันได้ ในขณะที่ผู้ใช้สามารถรัน model แบบ local ได้ฟรีทั้งหมด

---

## Problem Statement

อย่างที่กล่าวไปก่อนหน้านี้ การแปลงไฟล์ข้อมูลระหว่าง CSV, JSON และ TXT ในขั้นตอนการเตรียมข้อมูลมีปัญหาสำคัญคือไฟล์ที่ได้มาจากแต่ละแหล่งมักมีโครงสร้างและวิธีการจัดเรียงที่แตกต่างกันออกไป ทำให้ไม่มีกฎตายตัวเพียงชุดเดียวที่ครอบคลุมได้ทุกกรณี และในทางปฏิบัติการแปลงเหล่านี้ก็ไม่ได้ตรงไปตรงมาอย่างที่คิด ยกตัวอย่างเช่น

***ตัวอย่าง การแปลง CSV เป็นรูปแบบแบนราบ (flat / tabular) อย่างไม่ถูกต้อง***

[ภาพ: ตัวอย่างการแปลง File แบบ CSV เป็น JSON อย่างง่ายซึ่งอาจเกิดข้อผิดพลาดได้ — ประกอบจาก 01_csv_flat + 02_json_naive_wrong]

จากภาพข้างบนจะเห็นว่าเป็นการแปลงไฟล์ประเภท **CSV เป็นรูปแบบแบนราบ (flat / tabular csv)** ซึ่งหนึ่งแถวคือหนึ่งบรรทัด ไม่มีกลไกแสดงความสัมพันธ์แบบ "มีของซ้อนอยู่ข้างใน" ในขณะที่ JSON เป็นรูปแบบ **ลำดับชั้น (hierarchical / nested)** ที่ object มี array ซ้อนได้ การแปลง flat CSV เป็น JSON nested ที่ถูกต้องจึงไม่ใช่การ map ทีละแถว แต่ต้อง **คำนึงถึงโครงสร้างที่สัมพันธ์กันด้วย** ซึ่งจะทำให้ rule-based parser ทั่วไปทำพลาดได้ง่าย จึงต้องอาศัยการวิเคราะห์ในระดับหนึ่งเพื่อเขียน parser ที่ถูกต้องขึ้นมาได้:

1. **เรื่อง Grouping:** ต้องรู้ว่า `customer_id` คือ key ที่บอกว่าแถวไหนเป็นของ entity เดียวกัน แล้วยุบหลายแถวให้เหลือ object เดียวที่มี `orders[]` ถ้าวนทีละแถวเฉย ๆ wrapper จะหายและข้อมูลจะซ้ำ
2. **เรื่อง Type casting:** เพราะ CSV เก็บทุกค่าเป็น text ล้วน ๆ เราต้องรู้ว่า `amount` ควรเป็น number แล้วแปลงค่าอย่างถูกต้อง ไม่อย่างนั้นจะได้เลข `"250"` ที่เป็น text เปล่า ๆ นั่นเอง

❌ **โค้ดผิด: แปลงทีละแถวตรง ๆ (ผิด)**

```python
import csv
import json

records = []
with open("customers.csv", newline="") as f:
    for row in csv.DictReader(f):        # csv.DictReader คืนค่าทุกช่องเป็น str
        records.append({
            "customer_id": row["customer_id"],
            "name": row["name"],
            "amount": row["amount"],      # ✗ ไม่ cast → "250" ติดอยู่เป็น string
        })                                # ✗ 1 object ต่อ 1 row → ไม่ group ตาม customer_id

with open("output.json", "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
```

✅ **โค้ดถูก: จัดกลุ่มตาม customer แล้วแปลงค่า (ถูก)**

```python
import csv
import json
from collections import OrderedDict

customers = OrderedDict()                     # รักษาลำดับการเจอ customer ครั้งแรก
with open("customers.csv", newline="") as f:
    for row in csv.DictReader(f):
        cid = row["customer_id"]
        if cid not in customers:              # ✓ เจอ customer ใหม่จะสร้าง wrapper และ orders[]
            customers[cid] = {
                "customer_id": cid,
                "name": row["name"],
                "orders": [],
            }
        customers[cid]["orders"].append({     # ✓ row เดิมของ customer ต่อ array
            "order_id": row["order_id"],
            "amount": int(row["amount"]),     # ✓ แปลง str เป็น number
        })

records = list(customers.values())
with open("output.json", "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
```

***ตัวอย่าง การแปลง CSV รูปแบบแบนราบ (flat / tabular) ด้วย code ที่ถูกต้อง***

[ภาพ: ตัวอย่างการแปลง File แบบ CSV เป็น JSON อย่างง่ายด้วย Code ที่ถูกต้อง — 03_json_nested_correct]

จาก code ที่ผิดและถูกด้านบน จะเห็นได้ว่าในการแปลงไฟล์นี้มีจุดสำคัญที่ต้องคำนึงถึงสองอย่าง ซึ่งทั้งสองอย่างนี้เป็นความรู้ที่ **ไม่ได้เขียนอยู่ในไฟล์ตรง ๆ** ต้องอาศัยการเข้าใจ *บริบท* ของข้อมูล และนี่คือเหตุผลว่าทำไม rule-based ถึงไม่พอ ต้องใช้โมเดลที่ *"อ่านบริบทแล้วเขียน logic การแปลงได้เอง"*

**ปัญหาคือ ยิ่งไฟล์ซับซ้อนและใหญ่ขึ้น ปัญหายิ่งมากขึ้น:** ตัวอย่างนี้มีแค่ 3 แถว 1 ระดับการ group จึงดูเหมือนง่าย แต่ในไฟล์จริงความซับซ้อนมีมากกว่านั้นมาก ยกตัวอย่างเช่น

- **Grouping หลายชั้น:** ในความเป็นจริง group ที่มีซ้อนกันอาจไม่ใช่แค่ customer, orders แต่เป็น customer, orders, line, items, … การ group ผิดชั้นเดียวทำให้ข้อมูลทั้งชุดผิดตามกันหมด
- **Schema อาจไม่สม่ำเสมอ:** ข้อมูลบาง record อาจมี field ที่ record อื่น ๆ ไม่มี เช่น มีค่า null, วันที่หลายฟอร์แมตปนกัน หรือตัวเลขมี comma คั่นหลักพันในบางอันไม่มี ทำให้การแปลงแบบเหมารวมจะพังเป็นจุด ๆ ที่หายากในไฟล์ใหญ่และจำนวน record เยอะ ๆ
- **ความผิดพลาดมองไม่เห็น:** ในไฟล์ 3 แถว ตาเห็นได้ทันทีว่า Alice ซ้ำ แต่ในไฟล์ 50,000 แถว object ที่ควร group แต่ไม่ได้ group จะปนอยู่เงียบ ๆ แล้วไปโผล่เป็น bug ปลายทางที่ debug ยาก

ด้วยความไม่สม่ำเสมอเหล่านี้ **rule-based parser** จึงไม่สามารถรองรับได้ทุกกรณี ทุกครั้งที่เจอไฟล์ที่มีโครงสร้างแตกต่างออกไป developer ก็ต้องกลับมาเขียนหรือแก้ script เพิ่มเพื่อรองรับ case ใหม่ ซึ่งทั้งกินเวลาและเกิด error ได้ง่าย

ในทางกลับกัน LLM สามารถจัดการกับงานที่ขึ้นกับบริบทเหล่านี้ได้ดี โดยเฉพาะเมื่อใช้งานผ่านเครื่องมือ **Agentic CLI** อย่าง Claude Code หรือ Codex CLI ที่ให้โมเดลอ่านไฟล์ เขียน script และตรวจผลลัพธ์ได้เองในรอบเดียว

แต่วิธีนี้ก็มีข้อจำกัดสำคัญ 3 อย่าง คือ

### 1. ต้นทุนสูง

ทั้งค่า API ที่คิดตาม token และข้อจำกัดของ subscription เมื่อต้องเรียกใช้งานซ้ำ ๆ ในระดับ scale ขนาดใหญ่

[ภาพ: การใช้งาน Token บน Platform OpenRouter]

เราทำการทดสอบใช้ model จาก OpenRouter 3 ตัวที่มีคนใช้งานมากที่สุด ได้แก่ GPT 5.5, Claude Opus 4.8 และ DeepSeek V4 Pro ผ่านทาง OpenCode CLI โดยสั่งให้ model แปลงไฟล์ขนาดเล็กที่มีขนาดไม่เกิน 1KB ด้วย prompt ต่อไปนี้:

````text
# File-conversion script generator (cost-probe prompt)

You are an expert data engineer. You are given the **full contents** of a source
`{src}` file. Your job is to write a single self-contained Python script that
converts it to `{tgt}`.

Task for this file: {task}

Requirements for the script:
- Read the source from a path given as `sys.argv[1]` and write the converted
  result to `sys.argv[2]`.
- Use only the Python 3.12 standard library (`csv`, `json`, `re`, etc.).
- Handle realistic edge cases for this conversion (missing/empty fields,
  quoting/escaping, type coercion, nested structures, malformed lines) so the
  same script would work on other files of the same kind — not just this sample.
- Produce valid, well-formed `{tgt}` output.

Respond with your reasoning, then the script inside a single fenced code block:

```python
# your script here
```

Here is the full content of the source `{src}` file:

----- BEGIN FILE -----
{content}
----- END FILE -----
````

ได้ผลลัพธ์ทั้ง 3 Model ออกมาดังนี้

[ภาพ: ราคาค่า API จาก OpenRouter ต่อการแปลงไฟล์ 1 ไฟล์]

โดย model ทั้ง 3 ตัวสามารถแปลงไฟล์ได้อย่างถูกต้อง แต่หากว่าจะให้ model ระดับนี้ในการแปลงไฟล์ในระบบที่ขนาดใหญ่มากขึ้นก็จะส่งผลให้ cost ของการใช้งาน Model เพิ่มขึ้นตาม 3 อย่างนี้

[ภาพ: Input Token]

[ภาพ: Reasoning Token]

[ภาพ: Output Token]

1. **Input Token:** ทุก ๆ ครั้งที่คุณสั่งให้โมเดลทำการแปลงไฟล์ 1 ไฟล์ โมเดลจำเป็นที่จะต้องอ่านไฟล์ก่อนเพื่อที่จะรู้รูปแบบและโครงสร้างของไฟล์ก่อนที่จะทำการวางแผนและเขียน scripts
2. **Reasoning Token:** นี่คือส่วนที่ AI Model วางแผนในการเขียนสคริปต์ก่อนที่จะลงมือทำจริง ๆ โดยการคิดถึง edge case และขั้นตอนการแก้ปัญหาอย่างเป็นระบบ ขั้นตอนนี้ช่วยให้โมเดลผิดพลาดน้อยลงแต่ก็ทำให้ใช้ต้นทุนมากขึ้นเช่นกัน
3. **Output Token:** ตรงนี้คือส่วนที่โมเดลเขียนสคริปต์ออกมาเพื่อทำการแปลงไฟล์จริง ๆ

ซึ่งต้นทุนตรงนี้จะเริ่มเป็นปัญหามากขึ้นเมื่อเรายังคง prompt สั่ง AI Model ผ่าน Agentic CLI แบบปกติ เพราะโมเดลจำเป็นที่จะต้องอ่านไฟล์ทุกไฟล์อย่างละเอียดก่อนที่จะเริ่มทำการวางแผนเขียนสคริปต์ในการแปลงไฟล์ทั้งหมด เพื่อการันตีว่าจะไม่มี edge case และ error เกิดขึ้นในข้อมูลสุดท้ายนั่นเอง

[ภาพ: กราฟเปรียบเทียบต้นทุนที่ใช้ในการแปลง File ตามจำนวน]

### 2. ปัญหา Context window เต็ม

ถ้าใช้ LLM อ่านทั้งไฟล์ตรง ๆ ไฟล์ใหญ่จะทำให้ context ล้นหรือเกิด hallucination จน output เพี้ยนได้

### 3. ความเป็นส่วนตัว

ไฟล์ที่โมเดลอ่านในขณะทำงานอาจมีเนื้อหาที่ sensitive หรือมีข้อมูลส่วนตัวของผู้ใช้อยู่ ซึ่งบริษัทเจ้าของโมเดลอาจเก็บไปใช้ train model ซึ่งเป็นการละเมิดความเป็นส่วนตัวโดยที่ผู้ใช้อาจไม่รู้ตัวด้วยซ้ำ

---

## Pipeline Design & Solution

ด้วยเหตุผลและปัญหาทั้งหมด โครงงานนี้จึงต้องการพัฒนาเครื่องมือที่ทำงานได้แบบเดียวกันกับ **Agentic CLI** ได้ โดยที่ผู้ใช้สามารถรัน model แบบ local บนเครื่องตัวเองได้ฟรีทั้งหมด โดยกำหนด scope การแปลงไว้ที่ **5 Use Cases** ได้แก่

- **UC1:** CSV to JSON (nested)
- **UC2:** JSON to CSV (flattening)
- **UC3:** TXT log to CSV
- **UC4:** CSV to TXT (report)
- **UC5:** Schema migration

และทำการออกแบบโครงสร้างของระบบที่จะมาแก้ปัญหาในเรื่องของต้นทุนสูงและปัญหา context window เต็ม หัวใจของการออกแบบคือการเปลี่ยนวิธีการจากการ "ป้อนไฟล์ทั้งไฟล์ให้โมเดลอ่าน" มาเป็น "model ทำงานกับเครื่องมือ" แยกเครื่องมือออกจาก model อย่างชัดเจน โดยโมเดลไม่ต้องอ่านไฟล์ทั้งไฟล์อีกต่อไป แต่ทำงานกับ **metadata envelope** ขนาดเล็กที่จะป้อนข้อมูลที่จำเป็นให้กับ model แล้ว model จะทำหน้าที่เขียน Python script สั้น ๆ ส่งให้ sandbox รันบนไฟล์เต็มอีกที

จึงได้ออกแบบมาเป็น Pipeline 5 ขั้นตอนตามนี้

1. **Metadata Extractor:** ทำหน้าที่สกัด schema, types และ samples
2. **Context Summarizer:** ใช้ Gemma 2B base สร้างสรุปสำหรับแต่ละไฟล์สั้น ๆ อธิบายจุดประสงค์และโครงสร้างคร่าว ๆ
3. **Script Generator:** model จะรับเอาข้อมูลจาก stage ก่อนหน้ามาเขียน Python script ออกมาก่อนส่งต่อให้ขั้นตอนถัดไป
4. **Sandbox Executor:** รัน script ที่ model ทำบนไฟล์เต็มภายใต้ timeout
5. **Validator:** วัด 4 metrics (format, schema, load, content)

[ภาพ: แสดงโครงสร้างและ Flow ในการทำงานของ Pipeline]

โดยรายละเอียดการพัฒนาในแต่ละส่วนมีดังนี้

**1. Metadata Extractor** — เป็น stage แรกที่ทำงานแบบ deterministic ล้วน ๆ (ไม่มี model เข้ามาเกี่ยว) หน้าที่ของมันคืออ่านไฟล์ต้นทางแล้วสกัด metadata envelope ขนาดเล็กออกมา ประกอบด้วย schema, ชนิดข้อมูลของแต่ละ field (type inference), ตัวอย่างข้อมูลบางส่วน (samples) และ warnings ที่ฝัง domain knowledge ไว้ เช่น เตือนว่าคอลัมน์นี้น่าจะเป็น key สำหรับ group หรือเตือนว่าค่าตัวเลขถูกเก็บเป็น string เรารองรับครบทั้ง 3 format (CSV, JSON, TXT) โดยเขียน test แบบ end-to-end กำกับทุกตัว envelope นี้แหละคือสิ่งเดียวที่ model จะได้เห็น แทนที่จะเห็นไฟล์ทั้งไฟล์

**2. Context Summarizer** — ใช้ Gemma 2B base อ่าน metadata envelope แล้วสรุปเป็นภาษาธรรมชาติสั้น ๆ ว่าไฟล์นี้คืออะไร มีโครงสร้างแบบไหน และน่าจะต้องแปลงอย่างไร เป็นการเตรียม context ให้ stage ถัดไปเข้าใจงานก่อนลงมือเขียน script

**3. Script Generator** — เป็นหัวใจของ pipeline ตอน training เราใช้ Claude Opus เป็นคนเขียน Python script เพื่อเก็บเป็น training data แต่ตอน inference จริงเราสลับมาใช้ Gemma 4 2B ที่ fine-tune แล้วแทน script ที่ได้จะสั้น ๆ ประมาณ 30–50 บรรทัด และเป็น artifact ที่อ่านได้ debug ได้ ไม่ใช่ output ที่เป็น black box

**4. Sandbox Executor** — รัน script ที่ model เขียนบนไฟล์เต็มจริง ๆ ภายใต้ timeout เพื่อความปลอดภัย ขั้นตอนนี้ deterministic เช่นกัน จุดสำคัญคือ model ไม่เคยแตะไฟล์เต็มเลย มันแค่เขียน logic ส่วนการประมวลผลข้อมูลจริงเป็นหน้าที่ของ sandbox ทำให้ pipeline รองรับไฟล์ขนาดใหญ่เพียงใดก็ได้

**5. Validator** — ตรวจผลลัพธ์ด้วย 4 metrics (format, schema, load, content) และมีโหมด retry ≤ 3 ที่ป้อน error จาก sandbox กลับเข้า model ให้ลองแก้ใหม่ ซึ่งเราใช้โหมดนี้เป็น default ตอน production จริง

---

## Metric & Baseline

เมื่อได้โจทย์ที่ต้องการแก้แล้ว ขั้นตอนต่อมาคือการสำรวจว่าปัจจุบันปัญหานี้ถูกแก้ได้ดีเพียงใด ซึ่งจะกลายเป็น **baseline** สำหรับใช้เปรียบเทียบประสิทธิภาพของโมเดลที่พัฒนาขึ้น เราตัดสินใจใช้ **Gemma 4 2B** เป็น student model เพราะเป็น model ที่ขนาดเล็กและเหมาะกับการ run แบบ local

โดย metric ที่ทำการกำหนดสำหรับวัดผลการแปลงไฟล์ มีดังนี้

- **Format Validity** — ความถูกต้องของ format ปลายทาง
- **Schema Compliance** — โครงสร้างข้อมูลตรงตามที่กำหนดหรือไม่
- **Loadability** — โหลดไฟล์ผลลัพธ์ด้วย script ได้หรือไม่
- **Content Accuracy** — content ที่ map ไปถูกต้องครบถ้วนหรือไม่

ทั้ง 4 metrics ถูกเขียนเป็น functions แบบ module file พร้อม unit test จำนวน 28 ตัว ที่ผ่านทั้งหมด เพื่อให้การวัดผลมีความน่าเชื่อถือและทำซ้ำได้

หลังจากกำหนด metric เสร็จ เราวัด baseline ของ **Claude Opus** ในฐานะ teacher โดยให้รันผ่าน 5-stage pipeline เดียวกับที่ student จะใช้จริง (envelope → script → sandbox → validate) ซึ่งเป็นรอบเดียวกับตอนที่เก็บ training pairs ทั้ง 800 คู่ ทำให้เป็นการเปรียบเทียบแบบ apples-to-apples กับ student model ส่วน test set แบบ hand-crafted ที่ครอบคลุมทั้ง 5 Use Cases นั้นเราเก็บไว้ใช้ evaluate ตัว student ในขั้นถัดไป

[ภาพ: Baseline scores by use case + per-case heatmap]

ผลรวม baseline ของ Claude Opus เมื่อทำงานผ่าน pipeline ออกมาดังนี้ — Format Validity = 1.000, Schema Compliance = 1.000, Loadability = 1.000 และ Content Accuracy = 1.000 โดยทั้ง 800 คู่ผ่าน automated verification ครบ 100% (786 คู่ผ่านตั้งแต่ครั้งแรก มีแค่ 14 คู่ที่ต้อง retry 1 ครั้ง) ไม่มี error จาก teacher เลย

[ภาพ: Claude Opus baseline — overall scores]

จุดที่น่าสังเกตคือ การให้ teacher **เขียน script** แทนการแปลงไฟล์ตรง ๆ ช่วยปิดจุดอ่อนของ frontier model ที่เคยพลาดใน **UC1 (CSV → JSON nested)** ได้หมด — ทั้งเรื่องตัวเลขที่ถูกครอบด้วย quote กลายเป็น string และ wrapper ที่หายไปเมื่อมีหลาย row ต่อ entity เดียวกัน (ปัญหาที่เราเห็นในตัวอย่าง code ผิด/ถูกตอนต้นบทความ) เพราะ logic การ group และ cast ถูกเขียนลงใน script อย่างชัดเจน อย่างไรก็ตาม UC1 ยังเป็น case ที่เราจับตามากที่สุดตอนเก็บ training data เพราะเป็นงานที่ "เรียนรู้ยากที่สุด" — เดี๋ยวจะเห็นว่า student ตอนยังไม่ fine-tune ทำ UC1 ได้ 0/14

จากตัวเลข baseline นี้ เราตั้งเป้าหมายของ student model ไว้ว่าต้องทำได้ **≥ 80% ของ teacher ในทุก metric** ซึ่งเมื่อ teacher ได้ 1.000 ทุกตัว เป้าหมายจึงเท่ากับ **≥ 0.80 ทั้งสี่ metric** (FV / SC / LD / CA ≥ 0.80)

---

## Data Set

หลังจากได้ baseline แล้ว ขั้นตอนต่อมาคือการรวบรวมและสร้างชุดข้อมูลสำหรับ fine-tune

**แหล่งข้อมูล** — เรารวบรวมไฟล์จริงจากหลายแหล่ง ทั้ง Kaggle, Hugging Face และ GitHub โดยมี [data.go.th](https://data.go.th/) เป็นแหล่งสำคัญ เพราะมีไฟล์ CSV และ JSON ที่เป็นข้อมูลชุดเดียวกันแปลงไว้คู่กันอยู่แล้ว ทำให้เราใช้ verify ผลการแปลงได้ทันที

**การเก็บ training pairs** — เราใช้ Claude Opus เป็น teacher เก็บคู่ training ผ่าน 5-stage pipeline โดยทุกคู่ต้องผ่าน automated verification (เช่น `json.load`, `csv.reader`, schema check) ก่อนเข้า dataset สุดท้ายเก็บได้ครบ **800/800 คู่** ด้วย accept rate 100% โดย 786 คู่ผ่านตั้งแต่ครั้งแรก มีแค่ 14 คู่ที่ต้อง retry 1 ครั้ง จากนั้นแบ่งเป็น train / val / test = **650 / 80 / 70** และ verify ว่าทั้งสามชุด disjoint กัน มี 0 leakage ทุก example map กลับไปหา source ได้

แต่ตรงนี้หลายคนอาจสงสัยว่า *"แล้วโค้ดที่เก็บ dataset จริง ๆ มันทำงานยังไง?"* — เราขอลงรายละเอียดตรงนี้แบบ step-by-step พร้อม code จริงจาก repo เพราะหัวใจของ project นี้คือ **การกลั่นความรู้ของ teacher ออกมาเป็นชุดข้อมูล** ดังนั้น pipeline การเก็บ data จึงสำคัญพอ ๆ กับตัวโมเดล ภาพรวมของ loop การเก็บคู่ข้อมูลเป็นดังนี้ — แต่ละ case จะถูก teach → run → verify และรับเข้า dataset ต่อเมื่อผ่านเกณฑ์ ไม่อย่างนั้นก็ป้อน error กลับไปให้ teacher ลองใหม่:

[ภาพ: Teacher data-collection loop — extract → Opus writes script → sandbox → validate → accept/retry — data_collection_loop.png]

โดยทั้งหมดทำงานเป็น 5 ขั้น ดังนี้

***ขั้นที่ 0 — สร้าง synthetic corpus จาก generator ที่ seed ได้***

ก่อนจะเก็บ training pairs เราต้องมี "โจทย์" ให้ teacher ทำก่อน เราจึงเขียน generator แยกตาม Use Case (UC1–UC5) แต่ละตัวรับ `seed` กับ `complexity` แล้วสร้าง **ทั้งไฟล์ต้นทางและไฟล์เฉลย (expected output) จาก seed เดียวกัน** ทำให้ทุก case มี ground truth ไว้ verify ได้ทันที และ reproducible 100% (รันเมื่อไหร่ก็ได้ผลเดิม) ตัวอย่าง generator ของ UC1 (CSV แบน → JSON nested):

```python
# datamorph/data/generators/uc1_csv_to_json.py
def generate(seed: int, complexity: str) -> GeneratedCase:
    rng = random.Random(seed)
    fake = make_faker(seed)
    n_users = _N_USERS[complexity]          # simple/medium/complex คุมจำนวน entity
    max_orders = _MAX_ORDERS[complexity]    # และจำนวน order ต่อ user

    rows, records, oid = [], [], 1000
    for i in range(n_users):
        name = f"{fake.first_name()}{i}"     # index suffix กัน name ซ้ำ
        email = f"{name.lower()}@example.com"
        n_orders = 1 if complexity == "simple" else rng.randint(1, max_orders)
        orders = []
        for _ in range(n_orders):
            oid += 1
            item, price = rng.choice(_ITEMS), round(rng.uniform(1, 100), 2)
            rows.append((name, email, oid, item, price))     # ฝั่ง CSV (flat)
            orders.append({"id": oid, "item": item, "price": price})
        records.append({"user": {"name": name, "email": email}, "orders": orders})  # ฝั่ง JSON (nested)

    # input_text = CSV แบน, expected_text = JSON ที่ group แล้ว → คือเฉลยของ case นี้
    ...
    return GeneratedCase("uc1_csv_to_json_nested", complexity, "csv", "json",
                         input_text, expected_text, meta)
```

จุดสำคัญคือ `meta` ของแต่ละ case จะแนบ `prompt_hint` ที่อธิบายโจทย์ไว้ด้วย (เช่น *"group rows by user, keep order_price numeric"*) สั่ง `generate_corpus.py` ครั้งเดียวก็ได้ corpus ทั้งชุด พร้อม mix ความยากแบบ simple/medium/complex และ **กรอง hash ของ input ที่ชนกับ test set ทิ้งอัตโนมัติ** (กัน leakage ตั้งแต่ต้นน้ำ):

```bash
uv run python scripts/generate_corpus.py --count 800 --dest data/raw
```

***ขั้นที่ 1 — สกัด metadata envelope (ไม่มี model เข้ามาเกี่ยว)***

แต่ละ input file จะถูกส่งเข้า extractor ตาม format เพื่อสกัด envelope เล็ก ๆ ออกมา — นี่คือ **สิ่งเดียวที่ teacher จะได้เห็น** แทนที่จะเห็นไฟล์ทั้งไฟล์:

```python
# datamorph/data/envelope.py
def extract_envelope(input_path: Path, input_format: str) -> dict[str, Any]:
    """คืน metadata envelope ของ input_path โดย dispatch ตาม format (csv/json/txt)."""
    extractor_cls = _EXTRACTORS[input_format]
    return extractor_cls().extract(input_path)
```

***ขั้นที่ 2 — ให้ Claude Opus เขียน script จาก envelope***

เราประกอบ prompt จาก envelope + โจทย์ แล้วเรียก Opus ผ่าน CLI (`claude -p --model opus`) โดยบังคับให้ตอบกลับมาเป็น `<analysis>` + `<script>` เท่านั้น (อ่าน contract จาก Agent Skill ใน `skills/script_generation_teacher.md`) ที่สำคัญคือ prompt รับ `feedback` ได้ด้วย ใช้ตอน retry เพื่อบอก model ว่ารอบก่อนพังเพราะอะไร:

```python
# datamorph/data/teacher_script.py
def call_script_teacher(envelope, instruction, output_format, *, timeout=240, feedback=None):
    """รัน `claude -p --model opus` แล้ว parse <analysis> + <script> ออกมา."""
    prompt = build_script_prompt(envelope, instruction, output_format, feedback)
    cmd = ["claude", "-p", prompt, "--model", "opus",
           "--output-format", "json", "--allowedTools", "Read"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(PROJECT_ROOT), timeout=timeout,
                          encoding="utf-8", errors="replace")
    payload = json.loads(proc.stdout)             # claude -p คืน JSON พร้อม token usage
    analysis, script = parse_teacher_output(payload.get("result", ""))
    return ScriptResult(analysis, script, ..., payload)
```

***ขั้นที่ 3–4 — รัน script ใน sandbox แล้วให้คะแนน 4 metrics***

script ที่ Opus เขียนจะถูกรันใน subprocess sandbox (มี timeout + POSIX CPU limit) บนไฟล์เต็มจริง ๆ แล้วเอา output ไปเทียบกับเฉลยด้วย 4 metrics เกณฑ์รับคู่ค่อนข้างเข้ม — ต้องได้ format / schema / loadability เต็ม 1.0 และ content accuracy ≥ 0.95:

```python
# datamorph/data/collect.py
CA_MIN = 0.95

def _passes(scores: dict[str, float]) -> bool:
    return (scores.get("format_validity", 0.0) == 1.0
            and scores.get("loadability", 0.0) == 1.0
            and scores.get("schema_compliance", 0.0) == 1.0
            and scores.get("content_accuracy", 0.0) >= CA_MIN)
```

***ขั้นที่ 5 — loop เก็บคู่: teach → run → verify → retry ≤ 3***

ทุกอย่างถูกร้อยเข้าด้วยกันใน `collect_case()` ถ้า script ผ่านเกณฑ์ก็รับเข้า dataset ถ้าไม่ผ่าน (script พัง / คะแนนตก) จะป้อน `feedback` กลับเข้า teacher แล้วลองใหม่ได้สูงสุด 3 รอบ — กลไกนี้แหละที่ทำให้ accept rate แตะ 100% โดยมีแค่ 14 คู่ที่ต้อง retry:

```python
# datamorph/data/collect.py (ตัดมาเฉพาะ loop หลัก)
feedback = None
for attempt in range(max_retries + 1):
    tr = teacher_fn(envelope, instruction, output_format, feedback=feedback)  # Opus เขียน script
    if not tr.ok:
        feedback = "teacher produced no <script>"; continue

    sr = run_script(tr.script, input_path, output_suffix=out_ext)             # รันใน sandbox
    if not sr.ok:
        feedback = f"script {sr.error_kind}: {sr.stderr[:300]}"; continue     # syntax/runtime error → retry

    scores = score_all(actual=sr.output_text, expected=case.expected_text, ...)
    if _passes(scores):                                                       # ผ่านเกณฑ์ → รับเข้า dataset
        result.accepted = True
        return result
    feedback = f"output scored low: {scores}"                                 # คะแนนตก → retry พร้อมบอกว่าตกตรงไหน
```

คู่ที่ผ่านจะถูกเซฟเป็น JSON ใน `data/interim/` (เก็บครบทั้ง envelope + analysis + script + scores + token usage ของ Opus ไว้ทำ cost analysis ทีหลัง) สั่งทั้ง batch ด้วย:

```bash
uv run python scripts/collect_pairs.py --raw data/raw --interim data/interim
```

***ขั้นสุดท้าย — แปลงเป็น chat JSONL แล้ว split แบบ disjoint***

ก่อนเทรน เราแปลงทุก record ที่ verify แล้วให้อยู่ในรูป chat สองเทิร์น โดย **user message = envelope + โจทย์** (เหมือนกับตอน inference เป๊ะ ๆ — model ไม่เคยเห็นไฟล์เต็ม) และ **assistant message = `<analysis>` + `<script>`** ที่ผ่านการ verify มาแล้ว:

```python
# datamorph/features/format_pairs.py
def to_chat_record(record: dict[str, Any]) -> dict[str, Any]:
    env_json = json.dumps(record["envelope"], indent=2, default=str)
    user = (f"Metadata envelope:\n```json\n{env_json}\n```\n\n"
            f"Task: {record['instruction']}\n"
            f"Write a Python conversion script (reads sys.argv[1], writes sys.argv[2]; "
            f"stdlib + pandas only).")
    assistant = f"<analysis>{record['analysis']}</analysis>\n<script>\n{record['script']}\n</script>"
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}
```

ส่วนการ split train/val/test เราไม่ได้สุ่มแบบ random ธรรมดา แต่ใช้ **MD5 ของ `case_id` มา bucket** ทำให้ผลการแบ่ง deterministic (รันกี่ครั้งก็ได้ชุดเดิม) และ case เดียวกันจะตกใน split เดียวเสมอ — รับประกัน 0 leakage ระหว่างสามชุด:

```python
def _bucket(case_id: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{case_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF     # hash → ค่า [0,1) คงที่ข้ามเครื่อง

def split_records(records, *, val_frac=0.1, test_frac=0.1, seed=0):
    out = {"train": [], "val": [], "test": []}
    for rec in records:
        b = _bucket(rec["case_id"], seed)
        bucket = "test" if b < test_frac else "val" if b < test_frac + val_frac else "train"
        out[bucket].append(rec)
    return out
```

ผลสุดท้ายได้ไฟล์ `train.jsonl` / `val.jsonl` / `test.jsonl` ที่แต่ละบรรทัดหน้าตาแบบนี้ — พร้อมป้อนเข้า fine-tune ได้เลย:

```json
{
  "messages": [
    {"role": "user", "content": "Metadata envelope:\n```json\n{schema, samples, warnings}\n```\n\nTask: Convert CSV to JSON.\nWrite a Python conversion script..."},
    {"role": "assistant", "content": "<analysis>...</analysis>\n<script>\n...\n</script>"}
  ]
}
```

สรุป flow ทั้งหมดของการสร้าง dataset:

```
scripts/generate_corpus.py   →  data/raw/      (input + เฉลย จาก seed)
scripts/collect_pairs.py     →  data/interim/  (envelope→Opus→sandbox→verify→retry≤3)
scripts/build_dataset.py     →  data/processed/ (chat JSONL + split disjoint)
```

**EDA** — ก่อน fine-tune จริง เราพล็อตข้อมูลออกมาดูเพื่อเช็คความพร้อม พบว่า corpus สมดุลดี (160 คู่ต่อ Use Case แบ่งเป็น simple 400 / medium 280 / complex 120) ความยาว sequence มี median ~1,030 tokens สูงสุด ~2,094 tokens (100% ไม่เกิน 4,096) และ teacher ใช้ token ไปทั้งหมด ~1.69M output tokens (~2,116 tokens ต่อคู่)

[ภาพ: EDA — corpus balance / sequence length distribution]

**Base baseline ก่อน fine-tune** — เพื่อตอบคำถามว่า fine-tune จะต้องช่วยมากแค่ไหน เราเอา Gemma 4 2B base (ยังไม่ fine-tune) มารันบน pipeline เดียวกันก่อน ได้ผล **41/70** (fv 0.900, sc 0.771, ld 0.900, ca 0.657) เห็นชัดว่าจุดอ่อนหลักอยู่ที่ UC1 (CSV → nested JSON) ที่ได้ 0/14 และ UC3 ที่ได้ 6/14 — ระยะห่างจาก teacher ยังมาก แต่ยังไม่ถึงขั้นต้องเปลี่ยน base model แค่ต้อง fine-tune จริงจังด้วย training pairs ที่เก็บมา

[ภาพ: The four metrics — Teacher (Opus) vs Student base vs Student fine-tuned]

---

## Model fine-tune

**โมเดลตัวแรก** — เราเทรนด้วย LoRA SFT (rank 8 / alpha 16, lr 1e-4, 3 epochs = 1,950 iterations, train-on-completions) ได้ train loss 1.16 ใช้เวลาเทรนแค่ 30 นาที ผล evaluate บน test set 70 ไฟล์ออกมา **51/70** เพิ่มจาก base baseline (41/70) มา +10 ดูเผิน ๆ เหมือนดีขึ้น แต่พอดูราย Use Case กลับเจอปัญหา — **UC4 (CSV → report) พังหนัก** จาก 13/19 เหลือ 2/19 มี 17 cases ที่ crash ด้วย syntax/runtime error

[ภาพ: First model (3 epochs, no validation) — net +10 but uc4 collapsed]

**Error Analysis** — เราตั้งสมมติฐานและทดลองหา root cause พบว่า case UC4 ที่ fail ส่วนใหญ่เกิดจาก SyntaxError แบบเดียวกัน (วงเล็บ function ไม่ปิด) หรือ runtime error ตอน format report table สาเหตุคือ teacher script ของ UC4 ซับซ้อนที่สุดในบรรดา Use Case ทั้งหมด มีทั้ง `max()`, unpack, genexp และ ternary ที่เขียนเป็น multi-line list comprehension พอ student เลียนแบบ style มา แต่จัดการวงเล็บไม่ครบจึงเกิด error

เมื่อเราทำ **checkpoint sweep** (เซฟทุก 200 iterations) แล้วพล็อตคะแนน UC4 ออกมา ก็เจอ **inverted-U pattern** ชัดเจน — คะแนนพีคที่ iteration 400 แล้วค่อย ๆ ตกลงเพราะ overfit นี่ยืนยันว่า regression มาจากการเทรน 3 epochs โดยไม่มี validation set คอยหยุด ทำให้ checkpoint สุดท้ายเลยจุด optimum ไปไกลแล้ว

[ภาพ: Error analysis — uc4 is an inverted-U over training (peak at iter-400)]

---

## Final Model

**เลือก checkpoint** — เราเลือก checkpoint ที่ iteration 400 เป็น model หลัก ซึ่งทำได้ **65/70 (0.929 ทุก metric)** และชนะ base model ในทุก Use Case

[ภาพ: Per-use-case accuracy — before vs after fine-tune (iter-400)]

**Quantization** — เพื่อเตรียม deploy เราลอง quantize หลายระดับแล้วเทียบ accuracy กับขนาดไฟล์:

| Precision | Size | Accuracy |
|---|---:|:--:|
| bf16 | 9.6 GB | 65/70 |
| 8-bit | 5.5 GB | 64/70 |
| 4-bit | 4.1 GB | 58/70 |

เราเลือก **8-bit** เพราะเป็นจุดสมดุลที่สุด ส่วน 4-bit ทำให้ UC1 (nested JSON) พังจาก 14 เหลือ 9 เพราะ task นี้ sensitive ต่อ precision ของโมเดล

[ภาพ: Quantization trade-off — 8-bit is the accuracy/size pick]

**Retry mode** — เราเพิ่มโหมด retry ≤ 3 ที่ป้อน error จาก sandbox กลับเข้า model ให้ลองแก้ใหม่ และใช้เป็น default ใน production พอเปิดโหมดนี้ model 8-bit ทำได้ถึง **68/70 (0.971)** หรือประมาณ 97% ของ teacher

[ภาพ: Student accuracy journey on the 70-case held-out test set]

**Model Surgery** — ขั้นตอนสุดท้ายคือย่อขนาดโมเดลให้เล็กลงสำหรับ deploy เพราะ multimodal base ส่วนใหญ่เป็น dead weight สำหรับ task นี้ เราทำ surgery 4 ขั้น โดยจัดเรียงให้ขั้นที่ lossless (1–2) มาก่อน แล้วค่อยตามด้วยขั้นที่ลด precision (3–4) เพื่อพิสูจน์ก่อนว่า accuracy รอดในแต่ละจุด

[ภาพ: W7 model surgery — shrinking the student for deployment]

**(1) Fuse LoRA เข้า base model** — พับ adapter A/B กลับเข้า weight ของ language tower ตามสูตร `W += (alpha/rank) · (Bᵀ Aᵀ)` ซึ่งเทียบเท่ากับการรัน adapter ตอน inference เป๊ะ ๆ จึงไม่เสีย accuracy เลย (lossless):

```python
# scripts/build_textonly_student.py
def fuse_adapter(weights, adapter_file, rank, alpha):
    """Fold LoRA A/B into the base weights in place."""
    scale = alpha / rank                         # rank 8 / alpha 16 → scale = 2.0
    ad = _load_dict(str(adapter_file))
    targets = sorted({k[:-2] for k in ad if k.endswith(".A")})
    for t in targets:
        A = ad[t + ".A"]                         # (in, r)
        B = ad[t + ".B"]                         # (r, out)
        W = weights[t + ".weight"]               # (out, in)  base weight
        delta = scale * (B.T @ A.T)              # พับ adapter เป็น delta
        weights[t + ".weight"] = W + delta.astype(W.dtype)
    return len(targets), scale
```

**(2) Strip vision และ audio tower ทิ้ง** — task นี้ไม่เคยป้อน token รูปภาพ/เสียงเลย tower เหล่านี้จึงเป็น dead weight (~0.95 GB) เราเก็บเฉพาะ tensor ที่ขึ้นต้นด้วย `language_model.*` แล้ว rename ให้กลายเป็น `gemma4_text` ล้วน ๆ ที่ `mlx_lm` โหลดได้ตรง:

```python
# scripts/build_textonly_student.py
def to_text_only(weights, n_layers, n_kv_shared):
    """Keep only the language tower; rename language_model.model.X -> model.X."""
    out, dropped = {}, 0
    for k, v in weights.items():
        if not k.startswith("language_model."):
            dropped += 1                         # vision_tower / audio_tower / embed_* → ทิ้ง
            continue
        nk = k.replace("language_model.model.", "model.", 1)
        out[nk] = v
    return out, dropped
```

> ขั้น (1)+(2) รันด้วยคำสั่งเดียว:
>
> ```bash
> uv run python scripts/build_textonly_student.py \
>     --adapter models/lora_gemma4e2b_scriptgen/0000400_adapters.safetensors \
>     --out models/gemma4-e2b-textonly-iter400-bf16
> ```

**(3) Prune vocab จาก 262k เหลือ 16k** — embedding + PLE table เป็น tensor ที่ใหญ่ที่สุด และถูก index ด้วย vocab แต่ corpus ใช้จริงแค่ ~4.5k tokens เราจึงเก็บเฉพาะ token ที่ใช้ (+ merge closure + byte-fallback + special tokens) แล้ว **slice แถว embedding ที่ไม่ใช้ทิ้ง** โดยมี verification gate กันพัง — ถ้า tokenizer ที่ prune แล้ว re-segment ไม่ตรงของเดิมจะยกเลิกทันที ไม่แตะ weight:

```python
# scripts/prune_vocab.py
def slice_embeddings(model_dir, old2new, target):
    weights = mx.load(str(model_dir / "model.safetensors"))
    old_ids_in_new_order = sorted(old2new, key=lambda oid: old2new[oid])
    idx = mx.array(old_ids_in_new_order)
    for key in ("model.embed_tokens.weight", "model.embed_tokens_per_layer.weight"):
        w = weights[key]
        assert w.shape[0] == 262144            # ของเดิม 262k แถว
        weights[key] = w[idx]                  # เหลือเฉพาะแถวที่ keep → 16k
    return weights

# gate: re-tokenize ทั้ง corpus ด้วย tokenizer ที่ prune แล้ว ต้องได้ id เดิม (remap) เป๊ะ
def verify(orig_tok, pruned_tok, old2new, corpus_glob, extra_texts):
    for text in texts():
        orig = orig_tok.encode(text).ids
        want = [old2new[i] for i in orig if i in old2new]
        got = pruned_tok.encode(text).ids
        if got != want or pruned_tok.decode(got) != orig_tok.decode(orig):
            raise SystemExit("ABORT: segmentation not preserved")  # re-segment → ยกเลิก
```

> รันด้วย:
>
> ```bash
> uv run python scripts/prune_vocab.py \
>     --model models/gemma4-e2b-textonly-iter400-bf16 \
>     --target 16384 \
>     --out models/gemma4-e2b-textonly-iter400-bf16-vocab16k
> ```

**(4) Quantize 8-bit** — ขั้นสุดท้ายใช้ `mlx_lm.convert` ตรง ๆ (8-bit, group size 64) บนโมเดล text-only vocab-16k ได้ artifact สุดท้าย **2.0 GB** ที่ push ขึ้น HF Hub:

```bash
uv run python -m mlx_lm convert \
    --hf-path models/gemma4-e2b-textonly-iter400-bf16-vocab16k \
    --quantize --q-bits 8 --q-group-size 64 \
    --mlx-path models/data-morph-gemma-2b
```

สรุป chain ทั้ง 4 ขั้น: `iter-400 adapter` → **(1) fuse** → **(2) strip** → `bf16 text-only` → **(3) prune vocab** → `vocab-16k bf16 (3.8 GB)` → **(4) 8-bit** → `2.0 GB final ship artifact`

ผลลัพธ์สุดท้ายโมเดลเหลือเพียง **2.0 GB จาก 9.6 GB (−79%)** โดยยังทำได้ 67/70 (0.957) ในโหมด retry ≤ 3 ซึ่งสูงกว่าเป้าหมาย ≥ 80%-of-teacher ในทุก metric

| Artifact | Params | Size | retry ≤ 3 | % teacher |
|---|---:|---:|:--:|:--:|
| fine-tuned bf16 (runtime adapter) | 5.12 B | 9.6 GB | — | — |
| 8-bit (full model) | 5.1 B | 5.5 GB | 68/70 | ~97% |
| fused + text-only + vocab-16k, bf16 | 2.05 B | 3.8 GB | 69/70 (0.986) | ~99% |
| **+ 8-bit (final ship artifact)** | **2.05 B** | **2.0 GB** | **67/70 (0.957)** | **~96%** |

---

## Deployment

หลังได้โมเดลที่พอใจแล้ว เราทำการ deploy ดังนี้

- **Hugging Face Hub** — push โมเดล quantized ขนาด 2.0 GB พร้อม model card ขึ้น Hub
- **pip install-able wrapper** — wrap 5-stage pipeline (extractor → student script-gen → sandbox → validate, retry ≤ 3) ให้ติดตั้งและใช้งานได้จริง รองรับ Python 3.12.13
- **Hardware target** — primary คือ MacBook ด้วย MLX โดยมี fallback เป็น Google Colab + PyTorch + Unsloth

---

## Ethics

ในเรื่องของ AI Ethics เราคำนึงถึงประเด็นจริยธรรม 3 ข้อ ได้แก่

1. ไฟล์ที่แปลงอาจมีข้อมูลส่วนตัว ซึ่งจะไม่มีการ upload input ของผู้ใช้หรือนำไปใช้ต่อในการเทรน model เพราะทั้งหมด run แบบ local บน device ของผู้ใช้เอง
2. teacher bias สามารถส่งต่อไปยัง student model ได้ ซึ่งเราบันทึกข้อจำกัดนี้ไว้ใน model card อย่างชัดเจน เพื่อให้ผู้ใช้รับรู้ก่อนนำไปใช้งาน
3. ความเสี่ยงเรื่อง hallucination ถูกลดทอนด้วย automated format/schema validation ในขั้นตอน inference ทุกครั้ง ทำให้ output ที่ผิดรูปแบบถูกจับได้ก่อนส่งออก

---

Website page: <https://lovemig6334.github.io/data-morph/#data-morph>

PyPI Deploy: <https://pypi.org/project/data-morph-gemma/>

Repository: [github.com/LoveMig6334/data-morph](https://github.com/LoveMig6334/data-morph)
