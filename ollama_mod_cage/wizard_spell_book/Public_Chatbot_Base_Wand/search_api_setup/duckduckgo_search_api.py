"""Enhanced DuckDuckGo Search API with storage capabilities.
Handles text, image, and news searches with results persistence.
"""
from duckduckgo_search import DDGS, AsyncDDGS
import json
import aiohttp
import asyncio
from datetime import datetime
from pathlib import Path
import sqlite3
import logging
from typing import List, Dict, Optional, Union

class DuckDuckGoSearchAPI:
    def __init__(self, storage_type: str = "json", database_path: str = "search_results.db"):
        """
        Initialize the DuckDuckGo Search API wrapper.
        
        Args:
            storage_type: Type of storage ("json" or "sqlite")
            database_path: Path to the database file
        """
        self.storage_type = storage_type
        self.database_path = database_path
        self.setup_logging()
        
        if storage_type == "sqlite":
            self.setup_database()
    
    def setup_logging(self):
        """Configure logging for the API."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Initialize SQLite database with necessary tables."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS search_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT,
                        search_type TEXT,
                        timestamp DATETIME,
                        results TEXT
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database setup error: {e}")
            raise

    async def text_search(self, search_query: str, max_results: int = 10) -> List[Dict]:
        """
        Perform an async text search using DuckDuckGo.
        
        Args:
            search_query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of formatted search results
        """
        try:
            async with AsyncDDGS() as ddgs:
                results = await ddgs.text(
                    keywords=search_query,
                    region='wt-wt',
                    safesearch='off',
                    timelimit='y',
                    max_results=max_results
                )
                formatted_results = self.format_results(results)
                await self.store_results(search_query, "text", formatted_results)
                return formatted_results
        except Exception as e:
            self.logger.error(f"Text search error: {e}")
            raise

    async def image_search(self, search_query: str, max_results: int = 10) -> List[Dict]:
        """Perform an async image search using DuckDuckGo."""
        try:
            async with AsyncDDGS() as ddgs:
                results = await ddgs.images(
                    keywords=search_query,
                    region='wt-wt',
                    safesearch='off',
                    max_results=max_results
                )
                formatted_results = self.format_image_results(results)
                await self.store_results(search_query, "image", formatted_results)
                return formatted_results
        except Exception as e:
            self.logger.error(f"Image search error: {e}")
            raise

    async def news_search(self, search_query: str, max_results: int = 20) -> List[Dict]:
        """Perform an async news search using DuckDuckGo."""
        try:
            async with AsyncDDGS() as ddgs:
                results = await ddgs.news(
                    keywords=search_query,
                    region='wt-wt',
                    safesearch='off',
                    timelimit='m',
                    max_results=max_results
                )
                formatted_results = self.format_news_results(results)
                await self.store_results(search_query, "news", formatted_results)
                return formatted_results
        except Exception as e:
            self.logger.error(f"News search error: {e}")
            raise

    def format_results(self, results: List[Dict]) -> List[Dict]:
        """Format text search results."""
        return [{
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("body"),
            "source": result.get("source"),
            "timestamp": datetime.now().isoformat()
        } for result in results]

    def format_image_results(self, results: List[Dict]) -> List[Dict]:
        """Format image search results."""
        return [{
            "title": result.get("title"),
            "image_url": result.get("image"),
            "thumbnail": result.get("thumbnail"),
            "source_url": result.get("url"),
            "timestamp": datetime.now().isoformat()
        } for result in results]

    def format_news_results(self, results: List[Dict]) -> List[Dict]:
        """Format news search results."""
        return [{
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("excerpt"),
            "published": result.get("date"),
            "source": result.get("source"),
            "timestamp": datetime.now().isoformat()
        } for result in results]

    async def store_results(self, query: str, search_type: str, results: List[Dict]):
        """Store search results in the specified storage type."""
        if self.storage_type == "json":
            await self._store_json(query, search_type, results)
        elif self.storage_type == "sqlite":
            await self._store_sqlite(query, search_type, results)

    async def _store_json(self, query: str, search_type: str, results: List[Dict]):
        """Store results in a JSON file."""
        filename = f"search_results_{search_type}.json"
        try:
            existing_data = []
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            
            new_entry = {
                "query": query,
                "search_type": search_type,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            existing_data.append(new_entry)
            
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"JSON storage error: {e}")
            raise

    async def _store_sqlite(self, query: str, search_type: str, results: List[Dict]):
        """Store results in SQLite database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO search_results (query, search_type, timestamp, results)
                       VALUES (?, ?, ?, ?)""",
                    (query, search_type, datetime.now().isoformat(), json.dumps(results))
                )
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"SQLite storage error: {e}")
            raise

    async def get_stored_results(self, search_type: str, limit: int = 10) -> List[Dict]:
        """Retrieve stored results by search type."""
        if self.storage_type == "json":
            return await self._get_json_results(search_type, limit)
        elif self.storage_type == "sqlite":
            return await self._get_sqlite_results(search_type, limit)

    async def _get_json_results(self, search_type: str, limit: int) -> List[Dict]:
        """Retrieve results from JSON storage."""
        filename = f"search_results_{search_type}.json"
        try:
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    data = json.load(f)
                return data[-limit:]
            return []
        except Exception as e:
            self.logger.error(f"JSON retrieval error: {e}")
            raise

    async def _get_sqlite_results(self, search_type: str, limit: int) -> List[Dict]:
        """Retrieve results from SQLite storage."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT * FROM search_results 
                       WHERE search_type = ? 
                       ORDER BY timestamp DESC 
                       LIMIT ?""",
                    (search_type, limit)
                )
                results = cursor.fetchall()
                return [
                    {
                        "query": row[1],
                        "search_type": row[2],
                        "timestamp": row[3],
                        "results": json.loads(row[4])
                    }
                    for row in results
                ]
        except sqlite3.Error as e:
            self.logger.error(f"SQLite retrieval error: {e}")
            raise

# Example usage:
async def main():
    api = DuckDuckGoSearchAPI(storage_type="sqlite")
    
    # Perform searches
    text_results = await api.text_search("Python programming")
    image_results = await api.image_search("cute puppies")
    news_results = await api.news_search("technology")
    
    # Retrieve stored results
    stored_text_results = await api.get_stored_results("text")
    print(json.dumps(stored_text_results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())