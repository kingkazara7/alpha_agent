from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Core
    PORT: int = 8000
    
    # API Keys
    OPENAI_API_KEY: str
    DEEPSEEK_API_KEY: str
    GEMINI_API_KEY: str
    NOTION_API_KEY: str
    SEC_API_KEY: str
    ANTHROPIC_API_KEY: str
    
    # [NEW] Alpaca Keys
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    
    # Notion Database
    NOTION_DATABASE_ID: str
    
    class Config:
        env_file = ".env"
        # This tells Pydantic to ignore any extra keys in the .env file 
        # that aren't explicitly defined above, preventing future crashes.
        extra = "ignore" 

settings = Settings()