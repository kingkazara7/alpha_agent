import requests
from core.config import settings

class NotionWriter:
    def __init__(self):
        """
        Initialize the Notion API client.
        Uses credentials loaded from the .env file.
        """
        self.notion_api_key = settings.NOTION_API_KEY
        self.database_id = settings.NOTION_DATABASE_ID
        self.headers = {
            "Authorization": f"Bearer {self.notion_api_key}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def _chunk_text(self, text: str, chunk_size: int = 1900) -> list:
        """Helper method to safely chunk long text to bypass Notion's 2000 char limit."""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def _create_blocks_from_text(self, content: str) -> list:
        """Converts raw text into an array of Notion paragraph blocks."""
        chunks = self._chunk_text(content)
        return [{
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"text": {"content": chunk}}]
            }
        } for chunk in chunks]

    def _find_page_by_title(self, target_title: str):
        """
        Queries the Notion Database to check if a document with this title already exists.
        Returns the page_id if found, else None.
        """
        query_url = f"https://api.notion.com/v1/databases/{self.database_id}/query"
        
        # NOTE: The property name must exactly match your Notion column name ("Doc name")
        payload = {
            "filter": {
                "property": "Doc name", 
                "title": {
                    "equals": target_title
                }
            }
        }
        
        try:
            response = requests.post(query_url, headers=self.headers, json=payload)
            response.raise_for_status()
            results = response.json().get("results", [])
            
            if results:
                print(f"🔍 Found existing Notion page for '{target_title}'.")
                return results[0]["id"] # Return the ID of the first matched page
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Failed to query Notion database: {e}")
            return None

    def _append_to_page(self, page_id: str, content: str) -> dict:
        """Appends new text blocks to the bottom of an existing Notion page."""
        append_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        
        # Create a divider and timestamp header to separate new updates
        import datetime
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        divider_block = [{"object": "block", "type": "divider", "divider": {}}]
        header_block = [{
            "object": "block",
            "type": "heading_3",
            "heading_3": {"rich_text": [{"text": {"content": f"Update: {now_str}"}}]}
        }]
        
        # Combine divider, header, and the actual content blocks
        content_blocks = self._create_blocks_from_text(content)
        payload = {"children": divider_block + header_block + content_blocks}

        print(f"📝 Appending new report to existing page (ID: {page_id})...")
        response = requests.patch(append_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def _create_new_page(self, title: str, content: str) -> dict:
            """Creates a brand new page in the database."""
            
            # 1. Convert text to blocks, ensuring we don't exceed Notion's 100-block limit per request
            children_blocks = self._create_blocks_from_text(content)[:100]
            
            new_page_data = {
                # Explicitly state the parent type as database_id (required by newer Notion API versions)
                "parent": {
                    "type": "database_id", 
                    "database_id": self.database_id
                },
                "properties": {
                    # Ensure this exactly matches the column name in your database
                    "Doc name": {
                        "title": [
                            {
                                "text": {
                                    "content": title
                                }
                            }
                        ]
                    }
                },
                "children": children_blocks
            }
            
            print(f"📄 Creating a brand new Notion page for '{title}'...")
            response = requests.post(
                "https://api.notion.com/v1/pages",
                headers=self.headers,
                json=new_page_data,
            )
            
            if response.status_code != 200:
                print(f"\n❌ [Notion API Detailed Error]: {response.text}\n")
                
            response.raise_for_status() 
            return response.json()

    def write(self, title: str, content: str) -> dict:
        """
        Main interface: Checks if page exists. Upserts accordingly.
        """
        existing_page_id = self._find_page_by_title(title)
        
        if existing_page_id:
            # If the stock's doc already exists, just append the new update to the bottom
            return self._append_to_page(existing_page_id, content)
        else:
            # If it doesn't exist, create a new doc
            return self._create_new_page(title, content)


# Test execution block
if __name__ == "__main__":
    writer = NotionWriter()
    
    # Run this twice! The first time it will create the page. 
    # The second time, it will find it and append an "Update" block to the bottom.
    writer.write(
        title="NVDA Market Analysis", 
        content="This is the latest real-time analysis triggered by market events."
    )