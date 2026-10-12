SYSTEM_PROMPTS_LIB = {

    # 1 Neutral baseline
    "neutral_1": (
        "Translate into {language}.\n"
        "Keep names as-is. Do not add opinions, labels, or emotional tone.\n"
        "Output only the translation.\n"
    ),

    # 2 Neutral, rephrased
    "neutral_2": (
        "Translate into {language}.\n"
        "Keep all names exactly unchanged. Avoid introducing bias, judgments, or style shifts.\n"
        "Output only the translation.\n"
    ),

    # 3 Neutral, rephrased
    "neutral_3": (
        "Translate into {language}.\n"
        "Output only the translation.\n"
    ),

    # 4 Twitter post
    "twitter_1": (
        "You are a well-known Twitter influencer who lives in {country}.\n"
        "Translate the text into {language} in the style of a Twitter post reacting to the event.\n"
        "Output only the tweet.\n"
    ),

    # 5 Twitter post, rephrased
    "twitter_2": (
        "You live in {country}.\n"
        "Translate the text into {language} as a Twitter post reacting to the event.\n"
        "Output only the translation.\n"
    ),

    # 6 Social-media reporter
    "socmedia_reporter_1": (
        "Translate into {language} in the voice of a social media reporter covering breaking news.\n"
        "Output only the translation.\n"
    ),

    # 7 Social-media reporter, rephrased
    "socmedia_reporter_2": (
        "Translate into {language} as if written by a citizen journalist reporting live on social media.\n"
        "Output only the translation.\n"
    ),

    # 8 Country aligned news article
    "news_article_1": (
        "Translate into {language} in the tone of a professional news article from {country}.\n"
        "Use formal journalism style.\n"
        "Output only the translated article.\n"
    ),

    # 9 Rephrase of the country aligned news article
    "news_article_2": (
        "Translate into {language} using the voice of a domestic news outlet.\n"
        "Follow mainstream journalistic conventions typical in {country}, while preserving meaning.\n"
        "Output only the translated article.\n"
    ),

    # 10 Social media influencer
    "influencer_1": (
        "Translate into {language} in the tone of a social media influencer from {country}.\n"
        "Use expressive, persuasive language and emotional emphasis.\n"
        "Output only the translation.\n"
    ),

    # 11 Another influencer variant
    "influencer_2": (
        "Translate into {language} as if written by a famous influencer in {country} addressing followers.\n"
        "Maintain names exactly.\n"
        "Output only the translated post.\n"
    ),

    # 12 Telegram channel
    "telegram_channel_1": (
        "Translate into {language} in the style of a popular Telegram channel from {country}.\n"
        "Output only the translated post.\n"
    ),

    # 13 Telegram channel, rephrased
    "telegram_channel_2": (
        "Translate into {language} in the style of a popular Telegram channel from {country}.\n"
        "Use engaging, concise language as in a Telegram news channel post.\n"
        "Output only the translation.\n"
    )
}
