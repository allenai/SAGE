import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass


@dataclass
class ParsedContent:
    """Container for parsed website content"""

    url: str
    title: str
    main_content: str
    metadata: Dict[str, str]
    links: List[str]
    images: List[str]
    word_count: int
    content_type: str


def parse_website(
    url: str,
    timeout: int = 10,
    max_content_length: int = 50000,
    include_links: bool = False,
    include_images: bool = False,
    clean_content: bool = True,
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
) -> Optional[ParsedContent]:
    """
    Parse a website URL and extract content for RAG operations.

    Args:
        url: Website URL to parse
        timeout: Request timeout in seconds
        max_content_length: Maximum content length to extract
        include_links: Whether to extract internal links
        include_images: Whether to extract image URLs
        clean_content: Whether to clean and normalize content
        user_agent: User agent string for requests

    Returns:
        ParsedContent object or None if parsing fails
    """

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Referer": url,
        "Origin": urlparse(url).scheme + "://" + urlparse(url).netloc,
    }

    try:
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract title
        title = _extract_title(soup)

        # Extract main content
        main_content = _extract_main_content(soup, clean_content)

        # Truncate if too long
        if len(main_content) > max_content_length:
            main_content = main_content[:max_content_length] + "..."

        # Extract metadata
        metadata = _extract_metadata(soup)

        # Extract links
        links = _extract_links(soup, url) if include_links else []

        # Extract images
        images = _extract_images(soup, url) if include_images else []

        # Calculate word count
        word_count = len(main_content.split())

        # Determine content type
        content_type = _determine_content_type(soup, url)

        return ParsedContent(
            url=url,
            title=title,
            main_content=main_content,
            metadata=metadata,
            links=links,
            images=images,
            word_count=word_count,
            content_type=content_type,
        )

    except requests.RequestException as e:
        raise ValueError(f"Error fetching {url}: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing {url}: {e}")


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract page title"""
    # Try multiple title sources
    title_sources = [
        soup.find("title"),
        soup.find("h1"),
        soup.find("meta", {"property": "og:title"}),
        soup.find("meta", {"name": "title"}),
    ]

    for source in title_sources:
        if source:
            if source.name == "meta":
                title = source.get("content", "").strip()
            else:
                title = source.get_text().strip()

            if title:
                return title

    return "Untitled"


def _extract_main_content(soup: BeautifulSoup, clean_content: bool = True) -> str:
    """Extract main content from the page"""

    # Make a copy to avoid modifying original
    soup_copy = BeautifulSoup(str(soup), "html.parser")

    # Remove unwanted elements (but be less aggressive)
    unwanted_tags = ["script", "style", "noscript"]

    for tag in unwanted_tags:
        for element in soup_copy.find_all(tag):
            element.decompose()

    # Try to find main content container with site-specific selectors
    main_content_selectors = [
        # Wikipedia specific
        "#mw-content-text",
        ".mw-parser-output",
        "#bodyContent",
        # Generic selectors
        "main",
        "article",
        '[role="main"]',
        ".content",
        ".main-content",
        ".post-content",
        ".entry-content",
        ".article-content",
        "#content",
        "#main-content",
        ".page-content",
    ]

    main_element = None
    for selector in main_content_selectors:
        main_element = soup_copy.select_one(selector)
        if main_element:
            break

    # If no main content found, try paragraphs
    if not main_element:
        paragraphs = soup_copy.find_all("p")
        if paragraphs:
            # Create a temporary container
            main_element = soup_copy.new_tag("div")
            for p in paragraphs:
                main_element.append(p)

    # Last resort: use body
    if not main_element:
        main_element = soup_copy.find("body") or soup_copy

    # Remove specific unwanted elements after finding main content
    unwanted_in_content = [
        ".navbox",
        ".infobox",
        ".sidebar",
        ".toc",
        ".navigation-not-searchable",
        ".printfooter",
        ".catlinks",
        ".references",
        ".reflist",
        '[class*="nav"]',
        '[class*="menu"]',
        '[id*="nav"]',
        '[id*="menu"]',
    ]

    for selector in unwanted_in_content:
        for element in main_element.select(selector):
            element.decompose()

    # Extract text content with better spacing
    content_parts = []

    # Process each paragraph and heading separately for better structure
    for element in main_element.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
        text = element.get_text(strip=True)
        if text and len(text) > 10:  # Only include substantial text
            content_parts.append(text)

    # If no structured content found, fall back to all text
    if not content_parts:
        content = main_element.get_text(separator=" ", strip=True)
    else:
        content = "\n\n".join(content_parts)

    if clean_content:
        content = _clean_text_content(content)

    return content


def _clean_text_content(text: str) -> str:
    """Clean and normalize text content"""

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove excessive newlines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Remove common boilerplate text
    boilerplate_patterns = [
        r"cookies?\s+policy",
        r"privacy\s+policy",
        r"terms\s+of\s+service",
        r"subscribe\s+to\s+newsletter",
        r"follow\s+us\s+on",
        r"share\s+this\s+article",
        r"related\s+articles?",
        r"advertisement",
    ]

    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove URLs from text
    text = re.sub(r"https?://\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Clean up spacing again
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _extract_metadata(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract metadata from the page"""
    metadata = {}

    # Meta tags
    meta_tags = soup.find_all("meta")
    for tag in meta_tags:
        if tag.get("name"):
            metadata[tag["name"]] = tag.get("content", "")
        elif tag.get("property"):
            metadata[tag["property"]] = tag.get("content", "")

    # Canonical URL
    canonical = soup.find("link", {"rel": "canonical"})
    if canonical:
        metadata["canonical_url"] = canonical.get("href", "")

    # Language
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        metadata["language"] = html_tag["lang"]

    return metadata


def _extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract internal links from the page"""
    links = []
    base_domain = urlparse(base_url).netloc

    for link in soup.find_all("a", href=True):
        href = link["href"]

        # Convert relative URLs to absolute
        full_url = urljoin(base_url, href)

        # Only include internal links
        if urlparse(full_url).netloc == base_domain:
            links.append(full_url)

    return list(set(links))  # Remove duplicates


def _extract_images(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract image URLs from the page"""
    images = []

    for img in soup.find_all("img", src=True):
        src = img["src"]
        full_url = urljoin(base_url, src)
        images.append(full_url)

    return list(set(images))  # Remove duplicates


def _determine_content_type(soup: BeautifulSoup, url: str) -> str:
    """Determine the type of content on the page"""

    # Check for Wikipedia
    if "wikipedia.org" in url.lower():
        return "wikipedia_article"

    # Check for blog post indicators
    blog_indicators = [
        "article",
        ".post",
        ".blog-post",
        ".entry",
        "[datetime]",
        "time",
        ".author",
        ".byline",
    ]

    for indicator in blog_indicators:
        if soup.select(indicator):
            return "blog_post"

    # Check for documentation
    if any(word in url.lower() for word in ["docs", "documentation", "api", "guide"]):
        return "documentation"

    # Check for news article
    news_indicators = [
        '[property="article:published_time"]',
        ".news",
        ".article",
        '[itemtype*="NewsArticle"]',
    ]

    for indicator in news_indicators:
        if soup.select(indicator):
            return "news_article"

    # Check for product/service page
    if any(word in url.lower() for word in ["product", "service", "pricing", "buy"]):
        return "product_page"

    return "general_page"


# Example usage and testing
if __name__ == "__main__":
    # Test the function
    test_urls = [
        "https://www.formula1.com/en/latest/article/its-a-whole-different-mindset-how-haas-transformed-their-fortunes-as-they.46wHk7OMUm0dKkS2pWd1KA"
    ]

    for url in test_urls:
        print(f"\nParsing: {url}")
        result = parse_website(url)

        if result:
            print(f"Title: {result.title}")
            print(f"Content Type: {result.content_type}")
            print(f"Word Count: {result.word_count}")
            print(f"Content Preview: {result.main_content}...")
            print(f"Links Found: {len(result.links)}")
            print(f"Images Found: {len(result.images)}")
        else:
            print("Failed to parse website")