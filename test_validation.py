from headline_pipeline import validate_headline, post_process_headline, should_skip_article

def test_should_skip_article():
    # Length skip rule (> 15000 chars)
    assert should_skip_article("Original Title", "a" * 15001) == True
    assert should_skip_article("Original Title", "a" * 15000) == False
    
    # Title skip rules
    assert should_skip_article("Live Updates: election results", "content") == True
    assert should_skip_article("election results explained in detail", "content") == True
    assert should_skip_article("election guide from A to Z", "content") == True
    assert should_skip_article("Title | Segment 2 | Segment 3", "content") == True
    
    # Regular title and content should not skip
    assert should_skip_article("Normal Headline Title", "Short content") == False

def test_post_process_headline():
    # Trailing periods
    assert post_process_headline("This is a headline.") == "This is a headline"
    
    # Wrapping quotes
    assert post_process_headline('"This is a headline"') == "This is a headline"
    assert post_process_headline("'This is a headline'") == "This is a headline"
    assert post_process_headline('""This is a headline""') == "This is a headline"
    
    # Collapse whitespace
    assert post_process_headline("This   is   a   headline") == "This is a headline"
    
    # Sentence case first word, keep rest of casing
    assert post_process_headline("this is a headline about Modi") == "This is a headline about Modi"
    assert post_process_headline("BJP wins election") == "BJP wins election"

def test_validate_headline():
    title = "Coonoor commissioner caught taking bribe"
    content = "A Coonoor commissioner was caught taking a bribe of Rs 2 lakh today."
    
    # Valid headline
    valid, reason = validate_headline("Coonoor commissioner caught taking 2 lakh bribe", title, content)
    assert valid == True
    assert reason is None
    
    # Invalid proper noun (not in source)
    valid, reason = validate_headline("Mumbai commissioner caught taking bribe", title, content)
    assert valid == False
    assert "proper noun" in reason
    
    # Invalid number (not in source)
    valid, reason = validate_headline("Coonoor commissioner caught taking 5 lakh bribe", title, content)
    assert valid == False
    assert "number" in reason
    
    # First word proper-noun check validation
    # "Mumbai" is the first word and capitalized. It should fail since Mumbai is not in title/content.
    valid, reason = validate_headline("Mumbai catches corrupt commissioner", title, content)
    assert valid == False
    assert "proper noun 'Mumbai' not found" in reason
    
    # "Coonoor" as first word should succeed
    valid, reason = validate_headline("Coonoor officer arrested", title, content)
    assert valid == True
