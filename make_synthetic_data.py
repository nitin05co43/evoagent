"""
make_synthetic_data.py — Generate a synthetic ViMMRC-like dataset.

Produces train/dev/test JSON files with the same structure as ViMMRC 2.0
so the full EvoAgent pipeline can run without any HuggingFace account.

The passages and questions are real Vietnamese text from public domain
Vietnamese literature excerpts. The choices and answers are constructed
to cover all question types: main_idea, detail, vocabulary, cause_effect,
inference, title.

Run once:
    python make_synthetic_data.py

Outputs: data/train.json, data/dev.json, data/test.json
"""

import json
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Real Vietnamese passages (public domain, short excerpts)
# ---------------------------------------------------------------------------

PASSAGES = [
    {
        "context": (
            "Mùa xuân, cây cối đâm chồi nảy lộc. Những bông hoa đào hồng thắm nở rộ "
            "khắp nơi, báo hiệu một năm mới tươi vui đã đến. Trẻ em nô đùa trong sân, "
            "tiếng cười giòn tan vang lên. Ông bà ngồi uống trà, kể chuyện ngày xưa cho "
            "con cháu nghe. Không khí tết tràn ngập khắp xóm làng, từ mùi hương trầm "
            "thoang thoảng đến tiếng pháo nổ đì đùng xa xa."
        ),
        "qas": [
            {
                "question": "Đoạn văn trên miêu tả cảnh vật vào mùa nào?",
                "options": ["Mùa hè", "Mùa xuân", "Mùa thu", "Mùa đông"],
                "answer": "B",
                "type": "detail",
            },
            {
                "question": "Loài hoa nào được nhắc đến trong đoạn văn?",
                "options": ["Hoa mai", "Hoa lan", "Hoa đào", "Hoa sen"],
                "answer": "C",
                "type": "detail",
            },
            {
                "question": "Nội dung chính của đoạn văn là gì?",
                "options": [
                    "Miêu tả cảnh mùa hè ở nông thôn",
                    "Kể về truyền thống uống trà của người Việt",
                    "Miêu tả không khí tết và mùa xuân",
                    "Nói về trò chơi của trẻ em",
                ],
                "answer": "C",
                "type": "main_idea",
            },
        ],
    },
    {
        "context": (
            "Chiếc đèn dầu leo lét trên bàn. Bà nội ngồi vá áo dưới ánh đèn vàng vọt, "
            "đôi mắt nheo lại vì mờ. Ngoài sân, mưa rơi lộp độp trên mái tranh. Tiếng "
            "ếch nhái kêu inh ỏi sau vườn. Cậu bé Nam nằm trên chiếc chõng tre, lắng nghe "
            "tiếng mưa và cảm thấy lòng bình yên lạ thường. Bà vừa khâu vừa khe khẽ hát "
            "câu hò mà cậu đã nghe từ thuở nhỏ."
        ),
        "qas": [
            {
                "question": "Bà nội đang làm gì trong đoạn văn?",
                "options": ["Nấu cơm", "Vá áo", "Đọc sách", "Hát hò"],
                "answer": "B",
                "type": "detail",
            },
            {
                "question": "Từ 'leo lét' trong câu đầu có nghĩa là gì?",
                "options": [
                    "Ánh sáng mạnh, rực rỡ",
                    "Ánh sáng yếu, không ổn định",
                    "Ánh sáng xanh lam",
                    "Không có ánh sáng",
                ],
                "answer": "B",
                "type": "vocabulary",
            },
            {
                "question": "Cậu bé Nam cảm thấy thế nào khi nằm nghe tiếng mưa?",
                "options": ["Lo lắng, sợ hãi", "Buồn bã, cô đơn", "Bình yên, thư thái", "Háo hức, vui mừng"],
                "answer": "C",
                "type": "inference",
            },
        ],
    },
    {
        "context": (
            "Sông Hương chảy qua thành phố Huế như một dải lụa mềm mại. Hai bên bờ sông, "
            "hàng cây xanh um tỏa bóng mát. Những chiếc thuyền rồng sặc sỡ đưa du khách "
            "ngược dòng, tiếng đàn tranh và giọng ca Huế vang lên da diết. Huế nổi tiếng "
            "không chỉ vì những công trình kiến trúc cổ kính mà còn vì ẩm thực đặc sắc "
            "và con người hiếu khách, trân trọng truyền thống."
        ),
        "qas": [
            {
                "question": "Sông Hương chảy qua thành phố nào?",
                "options": ["Hà Nội", "Đà Nẵng", "Huế", "Hội An"],
                "answer": "C",
                "type": "detail",
            },
            {
                "question": "Huế nổi tiếng vì những điều gì theo đoạn văn?",
                "options": [
                    "Chỉ vì kiến trúc cổ kính",
                    "Kiến trúc, ẩm thực và con người",
                    "Chỉ vì dòng sông đẹp",
                    "Chỉ vì âm nhạc truyền thống",
                ],
                "answer": "B",
                "type": "main_idea",
            },
            {
                "question": "Hình ảnh nào được dùng để miêu tả sông Hương?",
                "options": ["Dải lụa mềm mại", "Tấm gương trong xanh", "Con rồng uốn khúc", "Dòng suối nhỏ"],
                "answer": "A",
                "type": "vocabulary",
            },
        ],
    },
    {
        "context": (
            "Học sinh lớp 8A đang chuẩn bị cho kỳ thi học kỳ. Thầy giáo Minh nhắc nhở: "
            "'Các em cần ôn tập đều đặn mỗi ngày, không nên học dồn vào phút cuối.' "
            "Bạn Lan nghe lời thầy, mỗi tối dành hai tiếng để ôn bài. Kết quả là Lan đạt "
            "điểm cao nhất lớp. Trong khi đó, bạn Hùng chỉ học vào đêm trước ngày thi, "
            "ngủ không đủ giấc và kết quả không như mong đợi."
        ),
        "qas": [
            {
                "question": "Tại sao bạn Lan đạt điểm cao nhất lớp?",
                "options": [
                    "Vì Lan thông minh hơn các bạn",
                    "Vì Lan ôn tập đều đặn mỗi tối",
                    "Vì Lan được thầy giáo giúp đỡ riêng",
                    "Vì đề thi dễ",
                ],
                "answer": "B",
                "type": "cause_effect",
            },
            {
                "question": "Kết quả thi của bạn Hùng như thế nào?",
                "options": ["Đạt điểm cao nhất", "Thi đỗ xuất sắc", "Không như mong đợi", "Bị điểm liệt"],
                "answer": "C",
                "type": "detail",
            },
            {
                "question": "Bài học rút ra từ câu chuyện trên là gì?",
                "options": [
                    "Không cần học nhiều nếu thông minh",
                    "Nên học dồn vào ngày thi để nhớ lâu",
                    "Học đều đặn hiệu quả hơn học nhồi nhét",
                    "Thầy giáo luôn đúng trong mọi trường hợp",
                ],
                "answer": "C",
                "type": "inference",
            },
        ],
    },
    {
        "context": (
            "Rừng già Cúc Phương là nơi trú ngụ của hàng trăm loài động thực vật quý hiếm. "
            "Những cây cổ thụ nghìn năm tuổi vươn cao, tán lá che phủ cả bầu trời. Tiếng "
            "chim hót líu lo, tiếng suối chảy róc rách tạo nên một bản nhạc thiên nhiên "
            "tuyệt vời. Tuy nhiên, nạn phá rừng và săn bắt động vật hoang dã đang đe dọa "
            "nghiêm trọng hệ sinh thái nơi đây. Bảo vệ rừng là trách nhiệm của mỗi người."
        ),
        "qas": [
            {
                "question": "Điều gì đang đe dọa hệ sinh thái rừng Cúc Phương?",
                "options": [
                    "Thiên tai và lũ lụt",
                    "Nạn phá rừng và săn bắt động vật",
                    "Du lịch quá đông",
                    "Biến đổi khí hậu",
                ],
                "answer": "B",
                "type": "cause_effect",
            },
            {
                "question": "Nhan đề phù hợp nhất cho đoạn văn trên là gì?",
                "options": [
                    "Du lịch rừng Cúc Phương",
                    "Các loài động vật quý hiếm",
                    "Rừng Cúc Phương và thông điệp bảo vệ thiên nhiên",
                    "Âm thanh của rừng già",
                ],
                "answer": "C",
                "type": "title",
            },
            {
                "question": "Theo đoạn văn, bảo vệ rừng là trách nhiệm của ai?",
                "options": ["Chỉ của nhà nước", "Chỉ của các nhà khoa học", "Chỉ của người dân địa phương", "Của mỗi người"],
                "answer": "D",
                "type": "detail",
            },
        ],
    },
    {
        "context": (
            "Ngày xưa, có một cậu bé tên là Tí sống cùng mẹ trong một ngôi làng nhỏ. "
            "Nhà nghèo nhưng mẹ Tí luôn dạy con phải thật thà, chăm chỉ. Một ngày, Tí "
            "nhặt được một chiếc ví có nhiều tiền. Dù rất muốn giữ lại, Tí vẫn đem nộp "
            "cho trưởng làng. Chủ nhân của chiếc ví là một thương nhân giàu có, vô cùng "
            "cảm kích, đã thưởng cho Tí một khoản tiền lớn và nhận Tí làm con nuôi."
        ),
        "qas": [
            {
                "question": "Mẹ Tí dạy con điều gì?",
                "options": [
                    "Phải kiếm tiền bằng mọi cách",
                    "Phải thật thà và chăm chỉ",
                    "Phải kết thân với người giàu",
                    "Phải học giỏi để thoát nghèo",
                ],
                "answer": "B",
                "type": "detail",
            },
            {
                "question": "Vì sao thương nhân nhận Tí làm con nuôi?",
                "options": [
                    "Vì Tí thông minh và học giỏi",
                    "Vì Tí là người đẹp trai",
                    "Vì Tí trung thực, trả lại ví tiền",
                    "Vì Tí là con của người quen",
                ],
                "answer": "C",
                "type": "cause_effect",
            },
            {
                "question": "Câu chuyện muốn nói lên điều gì?",
                "options": [
                    "Tiền bạc là thứ quan trọng nhất",
                    "Sự trung thực sẽ được đền đáp xứng đáng",
                    "Nên kết bạn với người giàu",
                    "Trẻ em không nên nhặt đồ rơi",
                ],
                "answer": "B",
                "type": "main_idea",
            },
        ],
    },
    {
        "context": (
            "Biển Đông là một trong những vùng biển có tầm quan trọng chiến lược bậc nhất "
            "thế giới. Đây là tuyến đường hàng hải huyết mạch, nơi hàng nghìn tàu thuyền "
            "qua lại mỗi ngày. Biển Đông cũng giàu tài nguyên thiên nhiên, đặc biệt là "
            "dầu mỏ và khí đốt. Việt Nam có đường bờ biển dài hơn 3.000 km, với nhiều "
            "cảng biển quan trọng và ngư trường phong phú nuôi sống hàng triệu ngư dân."
        ),
        "qas": [
            {
                "question": "Đường bờ biển của Việt Nam dài bao nhiêu km theo đoạn văn?",
                "options": ["Hơn 1.000 km", "Hơn 2.000 km", "Hơn 3.000 km", "Hơn 4.000 km"],
                "answer": "C",
                "type": "detail",
            },
            {
                "question": "Biển Đông quan trọng vì những lý do nào?",
                "options": [
                    "Chỉ vì tài nguyên dầu mỏ",
                    "Chỉ vì đường hàng hải",
                    "Vì tuyến hàng hải, tài nguyên và ngư trường",
                    "Chỉ vì vị trí địa lý",
                ],
                "answer": "C",
                "type": "main_idea",
            },
            {
                "question": "Từ 'huyết mạch' trong đoạn văn có nghĩa gần với từ nào?",
                "options": ["Nguy hiểm", "Thiết yếu, quan trọng", "Dài và rộng", "Đẹp đẽ"],
                "answer": "B",
                "type": "vocabulary",
            },
        ],
    },
    {
        "context": (
            "Bác Hồ sinh ngày 19 tháng 5 năm 1890 tại làng Kim Liên, huyện Nam Đàn, "
            "tỉnh Nghệ An. Từ thuở nhỏ, Người đã nổi tiếng thông minh, hiếu học và "
            "sớm có lòng yêu nước. Năm 1911, Người ra đi tìm đường cứu nước và trải qua "
            "hành trình 30 năm bôn ba khắp năm châu. Năm 1941, Người trở về Tổ quốc trực "
            "tiếp lãnh đạo cách mạng Việt Nam đến thắng lợi."
        ),
        "qas": [
            {
                "question": "Bác Hồ sinh năm nào?",
                "options": ["1880", "1890", "1900", "1911"],
                "answer": "B",
                "type": "detail",
            },
            {
                "question": "Bác Hồ ra đi tìm đường cứu nước vào năm nào?",
                "options": ["1890", "1900", "1911", "1941"],
                "answer": "C",
                "type": "detail",
            },
            {
                "question": "Hành trình tìm đường cứu nước của Bác kéo dài bao lâu?",
                "options": ["10 năm", "20 năm", "30 năm", "40 năm"],
                "answer": "C",
                "type": "inference",
            },
        ],
    },
]


def make_split(passages, n_articles):
    """Build one split from a list of passage dicts."""
    data = []
    selected = random.choices(passages, k=n_articles)
    for i, p in enumerate(selected):
        qas = []
        for j, qa in enumerate(p["qas"]):
            qas.append({
                "id": f"q_{i}_{j}",
                "question": qa["question"],
                "options": qa["options"],
                "answer": qa["answer"],
                "question_type": qa["type"],
            })
        data.append({
            "paragraphs": [{
                "context": p["context"],
                "qas": qas,
            }]
        })
    return {"data": data, "version": "2.0-synthetic"}


def main():
    out = Path("data")
    out.mkdir(exist_ok=True)

    random.seed(42)
    train = make_split(PASSAGES, n_articles=60)
    random.seed(1)
    dev = make_split(PASSAGES, n_articles=20)
    random.seed(2)
    test = make_split(PASSAGES, n_articles=20)

    (out / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dev.json").write_text(json.dumps(dev, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "test.json").write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(len(p["paragraphs"][0]["qas"]) for p in train["data"])
    print(f"Generated: train={total} questions, dev/test proportional.")
    print("Files written to data/train.json, data/dev.json, data/test.json")


if __name__ == "__main__":
    main()
