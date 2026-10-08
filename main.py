import feedparser
import smtplib
import json
import time
import os
from email.mime.text import MIMEText
from email.header import Header
from openai import OpenAI
from datetime import datetime, timedelta

# ================= 1. 环境与参数配置 =================
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.deepseek.com/v1")
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "deepseek-chat")

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "")

# 抓取最近 7 天（168 小时）文献，评分达到 6 分以上进入周报
TIME_WINDOW_HOURS = 168
MIN_AI_SCORE = 6

# 目标 7 大期刊的 RSS 订阅源
JOURNAL_FEEDS = {
    "JLT (Lightwave Tech)": "https://ieeexplore.ieee.org/rss/TOC50.XML",
    "PTL (Photonics Tech Lett)": "https://ieeexplore.ieee.org/rss/TOC68.XML",
    "OE (Optics Express)": "https://opg.optica.org/rss/oe.xml",
    "OL (Optics Letters)": "https://opg.optica.org/rss/ol.xml",
    "JOCN (J. Opt. Commun. Netw.)": "https://opg.optica.org/rss/jocn.xml",
    "NC (Nature Comms)": "https://www.nature.com/ncomms.rss",
    "NP (Nature Photonics)": "https://www.nature.com/nphoton.rss",
}

# 粗筛选关键词库（只要命中 1 个即送给 AI 精分析）
FILTER_KEYWORDS = [
    "mpi", "multi-path", "multipath", "crosstalk", "short-reach", "short reach",
    "equalization", "dsp", "volterra", "pam4", "direct detection", "coherent",
    "inter-symbol interference", "isi", "optical fiber communication", "im/dd"
]

# ================= 2. 功能逻辑 =================

def is_relevant_by_keywords(title, abstract):
    """正则/词库匹配预过滤"""
    content = (title + " " + abstract).lower()
    for kw in FILTER_KEYWORDS:
        if kw in content:
            return True, kw
    return False, None

def analyze_paper_with_ai(paper):
    """调用大模型评估与打分"""
    client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
    
    system_prompt = """
    你是一位光通信领域的顶尖专家。请评估以下文献与【短距高速光通信中 MPI（多径干扰/串扰）缓解算法设计】课题的相关性。
    请严格按照 JSON 格式输出，不要包含 Markdown 标记或多余文字：
    {
      "score": <0-10整数，10分代表直接研究MPI缓解算法>,
      "relevance_reason": "<一句话解释打分理由>",
      "innovation": "<提炼在算法、均衡器设计或系统传输上的核心创新点（50字内）>",
      "recommendation": "<精读 / 扫读 / 忽略>"
    }
    """
    
    user_content = f"期刊: {paper['journal']}\n标题: {paper['title']}\n摘要: {paper['abstract']}"
    
    try:
        response = client.chat.completions.create(
            model=AI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"AI 分析出现异常: {e}")
        return {"score": 0, "relevance_reason": "解析失败", "innovation": "无", "recommendation": "忽略"}

def send_weekly_email(high_score_papers):
    """发送 HTML 格式每周文献周报"""
    if not high_score_papers:
        print("本周无符合要求的高评分文献，跳过邮件发送。")
        return

    today_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"【文献周报】{today_str} 近一周光通信与 MPI 算法推荐 ({len(high_score_papers)}篇)"
    
    html_cards = ""
    for p in high_score_papers:
        ai = p['ai_eval']
        html_cards += f"""
        <div style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin-bottom: 16px; background-color: #ffffff;">
            <div style="font-size: 14px; color: #586069; font-weight: bold;">[{p['journal']}] <span style="color: #28a745; float: right;">AI 评分: {ai['score']}分 ({ai['recommendation']})</span></div>
            <h3 style="margin: 8px 0;"><a href="{p['link']}" style="color: #0366d6; text-decoration: none;">{p['title']}</a></h3>
            <p style="margin: 6px 0; font-size: 14px; color: #24292e;"><strong>💡 创新点：</strong>{ai['innovation']}</p>
            <p style="margin: 6px 0; font-size: 13px; color: #586069;"><strong>📌 分析依据：</strong>{ai['relevance_reason']}</p>
            <details style="margin-top: 8px; font-size: 12px; color: #6a737d;">
                <summary style="cursor: pointer;">展开查看摘要 (Abstract)</summary>
                <p style="margin-top: 6px; line-height: 1.4;">{p['abstract']}</p>
            </details>
        </div>
        """

    html_body = f"""
    <html>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; background-color: #f6f8fa; padding: 20px;">
        <div style="max-width: 700px; margin: 0 auto;">
            <h2 style="color: #24292e; border-bottom: 2px solid #e1e4e8; padding-bottom: 10px;">📚 最新光通信课题每周精选文献</h2>
            <p style="font-size: 14px; color: #586069;">过去 7 天内通过 RSS 扫描并经 AI 分析，为您挑选出以下高相关度文章：</p>
            {html_cards}
            <footer style="margin-top: 20px; text-align: center; font-size: 12px; color: #959da5;">
                自动推送系统 · GitHub Actions 驱动
            </footer>
        </div>
    </body>
    </html>
    """

    msg = MIMEText(html_body, 'html', 'utf-8')
    msg['From'] = Header("文献周报助手", 'utf-8')
    msg['To'] = Header(RECEIVER_EMAIL, 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')

    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg.as_string())
        server.quit()
        print("周报邮件已顺利发送！")
    except Exception as e:
        print(f"邮件发送失败: {e}")

# ================= 3. 执行入口 =================
if __name__ == "__main__":
    print(f"开始扫描过去 {TIME_WINDOW_HOURS} 小时（7天）的文献 - {datetime.now()}")
    time_threshold = datetime.now() - timedelta(hours=TIME_WINDOW_HOURS)
    matched_papers = []

    for journal_name, rss_url in JOURNAL_FEEDS.items():
        print(f"扫描期刊: {journal_name}")
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries:
            try:
                pub_time = datetime(*entry.published_parsed[:6])
                if pub_time < time_threshold:
                    continue
            except:
                pass

            title = entry.title
            abstract = entry.summary if hasattr(entry, 'summary') else ""
            link = entry.link

            # 1. 粗筛选
            hit, kw = is_relevant_by_keywords(title, abstract)
            if not hit:
                continue
            
            print(f" -> 命中关键词 [{kw}]，提交 AI 分析: {title[:30]}...")

            # 2. AI 评估
            ai_eval = analyze_paper_with_ai({"journal": journal_name, "title": title, "abstract": abstract})
            time.sleep(0.5)

            # 3. 收集高分文献
            if ai_eval.get("score", 0) >= MIN_AI_SCORE:
                matched_papers.append({
                    "journal": journal_name,
                    "title": title,
                    "link": link,
                    "abstract": abstract,
                    "ai_eval": ai_eval
                })

    print(f"检索完成，符合条件的优质文献共有 {len(matched_papers)} 篇。准备发送邮件...")
    send_weekly_email(matched_papers)