import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import random
from datetime import datetime, timedelta
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Set page configuration
def configure_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="InvestGenie - AI Portfolio Optimizer",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("InvestGenie - AI Portfolio Optimizer")
    st.markdown("""
    Welcome to InvestGenie! This application will help you create an optimized 
    investment portfolio based on your financial goals and risk tolerance.
    """)

def get_user_inputs():
    """
    Collect user inputs for portfolio optimization.
    
    Returns:
        tuple: Contains investment amount, risk tolerance, investment term, and AI features preferences
    """
    st.header("Your Investment Profile")
    
    with st.form("user_inputs"):
        col1, col2 = st.columns(2)
        
        with col1:
            investment_amount = st.number_input(
                "Investment Amount ($)",
                min_value=1000,
                max_value=10000000,
                value=10000,
                step=1000,
                help="Total amount you're planning to invest"
            )
            
            risk_tolerance = st.select_slider(
                "Risk Tolerance",
                options=["Low", "Medium-Low", "Medium", "Medium-High", "High"],
                value="Medium",
                help="Lower risk generally means more conservative investments with potentially lower returns"
            )
        
        with col2:
            investment_term = st.radio(
                "Investment Term",
                options=["Short-term (1-3 years)", "Medium-term (3-7 years)", "Long-term (7+ years)"],
                index=1,
                help="How long you plan to hold your investments"
            )
            
            include_crypto = st.checkbox(
                "Include Cryptocurrency",
                value=False,
                help="Whether to allocate a portion to cryptocurrency (higher risk)"
            )
        
        # AI Features section
        st.subheader("🤖 AI-Enhanced Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            use_sentiment_analysis = st.checkbox(
                "Include Sentiment Analysis",
                value=True,
                help="Analyze market sentiment for selected assets"
            )
        
        with col2:
            use_qa_analysis = st.checkbox(
                "Include Q&A Insights",
                value=True,
                help="Generate answers to common questions about your portfolio"
            )
            
        submitted = st.form_submit_button("Generate Portfolio")
        
        # Return values if the form is submitted
        if submitted:
            return investment_amount, risk_tolerance, investment_term, include_crypto, use_sentiment_analysis, use_qa_analysis
        else:
            return None, None, None, None, None, None

def fetch_financial_data(tickers, period="1y"):
    """
    Fetch financial data for the given tickers using yfinance.
    
    Args:
        tickers (list): List of ticker symbols
        period (str): Time period for historical data
    
    Returns:
        dict: Dictionary containing financial data for each ticker
    """
    financial_data = {}
    
    try:
        with st.spinner(f"Fetching data for {len(tickers)} assets..."):
            # Process in smaller batches to avoid API limitations
            batch_size = 5
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i+batch_size]
                
                for ticker in batch:
                    try:
                        # Fetch ticker data
                        ticker_data = yf.Ticker(ticker)
                        
                        # Get historical market data
                        hist = ticker_data.history(period=period)
                        
                        if not hist.empty:
                            # Calculate returns and volatility
                            returns = hist['Close'].pct_change().dropna()
                            annual_return = ((1 + returns.mean()) ** 252) - 1
                            annual_volatility = returns.std() * np.sqrt(252)
                            
                            # Get basic info
                            info = ticker_data.info
                            name = info.get('shortName', ticker)
                            
                            # Get recent news
                            try:
                                news = ticker_data.news[:3]  # Get latest 3 news items
                            except:
                                news = []
                            
                            # Store relevant data
                            financial_data[ticker] = {
                                'name': name,
                                'annual_return': annual_return,
                                'annual_volatility': annual_volatility,
                                'hist': hist,
                                'info': info,
                                'news': news
                            }
                        else:
                            st.warning(f"No data available for {ticker}")
                    except Exception as e:
                        st.warning(f"Error fetching data for {ticker}: {str(e)}")
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(0.5)
    
    except Exception as e:
        st.error(f"Failed to fetch financial data: {str(e)}")
    
    return financial_data

def get_asset_universe():
    """
    Define the universe of investment assets based on asset classes.
    
    Returns:
        dict: Dictionary of asset classes and their corresponding tickers
    """
    asset_universe = {
        'US_Large_Cap_Stocks': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG'],
        'US_Small_Cap_Stocks': ['IWM', 'AVUV', 'VB', 'IJR', 'DFAS'],
        'International_Stocks': ['VEA', 'VXUS', 'EFA', 'IEFA', 'VWO'],
        'Tech_Stocks': ['NVDA', 'TSLA', 'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'NFLX'],
        'Dividend_Stocks': ['VYM', 'SCHD', 'HDV', 'DVY', 'PFE', 'VZ', 'KO', 'MO'],
        'US_Bonds': ['AGG', 'BND', 'IEI', 'IEF', 'TLT', 'GOVT', 'MUB', 'LQD'],
        'International_Bonds': ['BNDX', 'IAGG', 'EMB', 'PCY'],
        'Inflation_Protected': ['TIP', 'SCHP', 'VTIP', 'STIP'],
        'Real_Estate': ['VNQ', 'SCHH', 'IYR', 'RWR', 'REET'],
        'Commodities': ['GLD', 'IAU', 'SLV', 'PDBC', 'GSG', 'BCI'],
        'ESG_Funds': ['ESGU', 'ESGV', 'ESGE', 'ESML', 'USSG'],
        'Target_Date_ETFs': ['VTTVX', 'VFIFX', 'VTTSX'],
        'Crypto_ETFs': ['BITO', 'BITW', 'GBTC', 'ETHE']
    }
    
    return asset_universe

def calculate_risk_profile(risk_tolerance, investment_term, include_crypto=False):
    """
    Calculate the asset allocation percentages based on risk tolerance and investment term.
    
    Args:
        risk_tolerance (str): User's risk tolerance level
        investment_term (str): User's investment term preference
        include_crypto (bool): Whether to include cryptocurrency allocation
    
    Returns:
        dict: Allocation percentages for each asset class
    """
    # Base allocations by risk tolerance
    risk_profiles = {
        "Low": {
            "US_Stocks": 20,
            "International_Stocks": 10,
            "Bonds": 60,
            "Alternatives": 10,
            "Crypto": 0
        },
        "Medium-Low": {
            "US_Stocks": 35,
            "International_Stocks": 15,
            "Bonds": 40,
            "Alternatives": 10,
            "Crypto": 0
        },
        "Medium": {
            "US_Stocks": 45,
            "International_Stocks": 20,
            "Bonds": 25,
            "Alternatives": 10,
            "Crypto": 0
        },
        "Medium-High": {
            "US_Stocks": 55,
            "International_Stocks": 25,
            "Bonds": 10,
            "Alternatives": 10,
            "Crypto": 0
        },
        "High": {
            "US_Stocks": 65,
            "International_Stocks": 25,
            "Bonds": 0,
            "Alternatives": 10,
            "Crypto": 0
        }
    }
    
    # Get base allocation based on risk tolerance
    allocation = risk_profiles[risk_tolerance].copy()
    
    # Adjust for investment term
    if "Short-term" in investment_term:
        # More conservative for short-term
        allocation["Bonds"] = min(allocation["Bonds"] + 10, 70)
        allocation["US_Stocks"] = max(allocation["US_Stocks"] - 5, 10)
        allocation["International_Stocks"] = max(allocation["International_Stocks"] - 5, 5)
    elif "Long-term" in investment_term:
        # More aggressive for long-term
        allocation["US_Stocks"] = min(allocation["US_Stocks"] + 5, 70)
        allocation["International_Stocks"] = min(allocation["International_Stocks"] + 5, 30)
        allocation["Bonds"] = max(allocation["Bonds"] - 10, 0)
    
    # Adjust for cryptocurrency if requested
    if include_crypto:
        # Take a small percentage from stocks and alternatives for crypto
        crypto_alloc = min(5, 15 - allocation["Alternatives"])
        stock_reduction = crypto_alloc // 2
        
        allocation["Crypto"] = crypto_alloc
        allocation["US_Stocks"] = max(allocation["US_Stocks"] - stock_reduction, 10)
        allocation["Alternatives"] = max(allocation["Alternatives"] - (crypto_alloc - stock_reduction), 5)
    
    return allocation

def get_detailed_allocation(risk_profile, asset_universe):
    """
    Create a detailed allocation of specific assets based on the risk profile.
    
    Args:
        risk_profile (dict): Asset class allocation percentages
        asset_universe (dict): Available assets by category
    
    Returns:
        dict: Detailed allocation of specific assets with their percentages
    """
    detailed_allocation = {}
    
    # US Stocks allocation
    us_stocks_total = risk_profile["US_Stocks"]
    if us_stocks_total > 0:
        # Split US stocks by market cap and style
        large_cap_pct = 0.6  # 60% to large cap
        small_cap_pct = 0.15  # 15% to small cap
        tech_pct = 0.15  # 15% to tech
        dividend_pct = 0.1  # 10% to dividend
        
        # Select tickers for each category
        large_cap_tickers = random.sample(asset_universe["US_Large_Cap_Stocks"], min(4, len(asset_universe["US_Large_Cap_Stocks"])))
        small_cap_tickers = random.sample(asset_universe["US_Small_Cap_Stocks"], min(2, len(asset_universe["US_Small_Cap_Stocks"])))
        tech_tickers = random.sample(asset_universe["Tech_Stocks"], min(3, len(asset_universe["Tech_Stocks"])))
        dividend_tickers = random.sample(asset_universe["Dividend_Stocks"], min(2, len(asset_universe["Dividend_Stocks"])))
        
        # Allocate percentages to individual tickers
        for ticker in large_cap_tickers:
            detailed_allocation[ticker] = us_stocks_total * large_cap_pct / len(large_cap_tickers)
            
        for ticker in small_cap_tickers:
            detailed_allocation[ticker] = us_stocks_total * small_cap_pct / len(small_cap_tickers)
            
        for ticker in tech_tickers:
            detailed_allocation[ticker] = us_stocks_total * tech_pct / len(tech_tickers)
            
        for ticker in dividend_tickers:
            detailed_allocation[ticker] = us_stocks_total * dividend_pct / len(dividend_tickers)
    
    # International Stocks allocation
    intl_stocks_total = risk_profile["International_Stocks"]
    if intl_stocks_total > 0:
        intl_tickers = random.sample(asset_universe["International_Stocks"], min(3, len(asset_universe["International_Stocks"])))
        for ticker in intl_tickers:
            detailed_allocation[ticker] = intl_stocks_total / len(intl_tickers)
    
    # Bonds allocation
    bonds_total = risk_profile["Bonds"]
    if bonds_total > 0:
        # Split bonds between US and international
        us_bonds_pct = 0.75  # 75% to US bonds
        intl_bonds_pct = 0.15  # 15% to international bonds
        inflation_pct = 0.10  # 10% to inflation protected
        
        us_bond_tickers = random.sample(asset_universe["US_Bonds"], min(3, len(asset_universe["US_Bonds"])))
        intl_bond_tickers = random.sample(asset_universe["International_Bonds"], min(2, len(asset_universe["International_Bonds"])))
        inflation_tickers = random.sample(asset_universe["Inflation_Protected"], min(1, len(asset_universe["Inflation_Protected"])))
        
        for ticker in us_bond_tickers:
            detailed_allocation[ticker] = bonds_total * us_bonds_pct / len(us_bond_tickers)
            
        for ticker in intl_bond_tickers:
            detailed_allocation[ticker] = bonds_total * intl_bonds_pct / len(intl_bond_tickers)
            
        for ticker in inflation_tickers:
            detailed_allocation[ticker] = bonds_total * inflation_pct / len(inflation_tickers)
    
    # Alternatives allocation
    alt_total = risk_profile["Alternatives"]
    if alt_total > 0:
        # Split alternatives between real estate and commodities
        real_estate_pct = 0.6  # 60% to real estate
        commodities_pct = 0.4  # 40% to commodities
        
        real_estate_tickers = random.sample(asset_universe["Real_Estate"], min(2, len(asset_universe["Real_Estate"])))
        commodities_tickers = random.sample(asset_universe["Commodities"], min(2, len(asset_universe["Commodities"])))
        
        for ticker in real_estate_tickers:
            detailed_allocation[ticker] = alt_total * real_estate_pct / len(real_estate_tickers)
            
        for ticker in commodities_tickers:
            detailed_allocation[ticker] = alt_total * commodities_pct / len(commodities_tickers)
    
    # Crypto allocation if applicable
    crypto_total = risk_profile.get("Crypto", 0)
    if crypto_total > 0:
        crypto_tickers = random.sample(asset_universe["Crypto_ETFs"], min(2, len(asset_universe["Crypto_ETFs"])))
        for ticker in crypto_tickers:
            detailed_allocation[ticker] = crypto_total / len(crypto_tickers)
    
    return detailed_allocation

def calculate_portfolio_metrics(detailed_allocation, financial_data, investment_amount):
    """
    Calculate portfolio metrics based on allocation and financial data.
    
    Args:
        detailed_allocation (dict): Asset allocation percentages
        financial_data (dict): Financial data for assets
        investment_amount (float): Total investment amount
    
    Returns:
        tuple: Portfolio performance metrics and detailed allocation with dollar amounts
    """
    portfolio_dollar_allocation = {}
    weighted_return = 0
    weighted_volatility = 0
    total_allocated = 0
    
    # Process each asset in the allocation
    for ticker, percentage in detailed_allocation.items():
        if ticker in financial_data:
            # Calculate dollar amount
            dollar_amount = investment_amount * (percentage / 100)
            
            # Get asset data
            asset_data = financial_data[ticker]
            annual_return = asset_data.get('annual_return', 0)
            annual_volatility = asset_data.get('annual_volatility', 0)
            name = asset_data.get('name', ticker)
            
            # Add to portfolio metrics
            weighted_return += annual_return * (percentage / 100)
            weighted_volatility += annual_volatility * (percentage / 100)
            total_allocated += percentage
            
            # Store detailed information
            portfolio_dollar_allocation[ticker] = {
                'name': name,
                'percentage': percentage,
                'dollar_amount': dollar_amount,
                'expected_return': annual_return,
                'volatility': annual_volatility
            }
    
    # Calculate expected portfolio performance
    portfolio_metrics = {
        'expected_annual_return': weighted_return,
        'expected_volatility': weighted_volatility,
        'sharpe_ratio': weighted_return / weighted_volatility if weighted_volatility > 0 else 0,
        'total_allocated': total_allocated
    }
    
    return portfolio_metrics, portfolio_dollar_allocation

def categorize_assets(portfolio_allocation, asset_universe):
    """
    Categorize assets in the portfolio into stocks, bonds, ETFs, etc.
    
    Args:
        portfolio_allocation (dict): Portfolio allocation
        asset_universe (dict): Asset universe categorization
    
    Returns:
        dict: Categorized portfolio
    """
    # Create reverse lookup from ticker to category
    ticker_to_category = {}
    for category, tickers in asset_universe.items():
        for ticker in tickers:
            ticker_to_category[ticker] = category
    
    # Initialize categories
    categories = {
        'Stocks': {},
        'Bonds': {},
        'ETFs': {},
        'Alternatives': {},
        'Crypto': {}
    }
    
    # Categorize each asset
    for ticker, details in portfolio_allocation.items():
        category = ticker_to_category.get(ticker, '')
        
        if category in ['US_Large_Cap_Stocks', 'US_Small_Cap_Stocks', 'Tech_Stocks', 'Dividend_Stocks']:
            categories['Stocks'][ticker] = details
        elif category in ['US_Bonds', 'International_Bonds', 'Inflation_Protected']:
            categories['Bonds'][ticker] = details
        elif category in ['International_Stocks', 'ESG_Funds', 'Target_Date_ETFs']:
            categories['ETFs'][ticker] = details
        elif category in ['Real_Estate', 'Commodities']:
            categories['Alternatives'][ticker] = details
        elif category in ['Crypto_ETFs']:
            categories['Crypto'][ticker] = details
        else:
            # Default to ETFs for anything not explicitly categorized
            categories['ETFs'][ticker] = details
    
    return categories

# New functions for GenAI integration

def analyze_portfolio_sentiment(portfolio_allocation, financial_data):
    """
    Analyze sentiment for the assets in the portfolio using FinBERT.
    
    Args:
        portfolio_allocation (dict): Portfolio allocation
        financial_data (dict): Financial data including news
    
    Returns:
        dict: Sentiment analysis results for each asset
    """
    sentiment_results = {}
    
    try:
        with st.spinner("Analyzing market sentiment with FinBERT..."):
            # Initialize the sentiment pipeline
            sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            
            # Create sample news if no news is available
            sample_news_templates = [
                "{ticker} has been showing steady performance in the current market conditions.",
                "Analysts are cautiously optimistic about {ticker}'s outlook for the next quarter.",
                "{ticker} reported earnings that met expectations despite challenging market conditions.",
                "Market experts are watching {ticker} closely as it navigates through current economic headwinds."
            ]
            
            # Analyze sentiment for each asset in the portfolio
            for ticker, details in portfolio_allocation.items():
                if ticker in financial_data:
                    # Get news for the ticker or use sample news
                    news_items = financial_data[ticker].get('news', [])
                    if not news_items:
                        # Use sample news if no news is available
                        news_text = random.choice(sample_news_templates).format(ticker=ticker)
                    else:
                        # Use the title from the first news item
                        news_text = news_items[0].get('title', f"News about {ticker}")
                    
                    # Run sentiment analysis
                    sentiment_result = sentiment_pipeline(news_text)
                    sentiment_label = sentiment_result[0]['label']
                    sentiment_score = sentiment_result[0]['score']
                    
                    # Store the results
                    sentiment_results[ticker] = {
                        'text': news_text,
                        'label': sentiment_label,
                        'score': sentiment_score
                    }
    
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        # Provide simulated results if there's an error
        for ticker in portfolio_allocation.keys():
            sentiment_results[ticker] = {
                'text': f"Simulated sentiment analysis for {ticker}",
                'label': random.choice(['positive', 'neutral', 'negative']),
                'score': random.uniform(0.6, 0.95)
            }
    
    return sentiment_results

def generate_portfolio_qa(portfolio_metrics, risk_tolerance, investment_term, include_crypto):
    """
    Generate Q&A insights about the portfolio using DistilBERT.
    
    Args:
        portfolio_metrics (dict): Portfolio performance metrics
        risk_tolerance (str): User's risk tolerance level
        investment_term (str): User's investment term preference
        include_crypto (bool): Whether cryptocurrency is included
    
    Returns:
        dict: Q&A insights about the portfolio
    """
    qa_insights = {}
    
    try:
        with st.spinner("Generating portfolio insights with DistilBERT..."):
            # Initialize the QA pipeline
            qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
            qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
            
            # Create context for the QA model based on portfolio characteristics
            expected_return = portfolio_metrics['expected_annual_return'] * 100
            volatility = portfolio_metrics['expected_volatility'] * 100
            sharpe_ratio = portfolio_metrics['sharpe_ratio']
            
            context = f"""
            This investment portfolio has an expected annual return of {expected_return:.2f}% with 
            a volatility of {volatility:.2f}%. The portfolio's Sharpe ratio is {sharpe_ratio:.2f}, 
            which is a measure of risk-adjusted return. The portfolio was constructed based on a {risk_tolerance.lower()} 
            risk tolerance and a {investment_term.lower()} investment horizon. 
            
            For a {risk_tolerance.lower()} risk tolerance, this portfolio has an appropriate balance of 
            stocks, bonds, and alternative investments. The higher the risk tolerance, the greater the 
            allocation to stocks and growth assets. A {risk_tolerance.lower()} risk investor prioritizes 
            {'capital preservation with some growth potential' if risk_tolerance in ['Low', 'Medium-Low'] else 'balanced growth and stability' if risk_tolerance == 'Medium' else 'growth potential over stability'}.
            
            The {investment_term.lower()} investment horizon allows for {'limited exposure to market volatility' if 'Short-term' in investment_term else 'moderate exposure to market cycles' if 'Medium-term' in investment_term else 'riding out market cycles for long-term growth'}.
            
            {'The portfolio includes some exposure to cryptocurrency as an alternative asset class, which adds potential for higher returns but also increases overall portfolio risk.' if include_crypto else 'The portfolio does not include cryptocurrency exposure, focusing instead on traditional asset classes.'}
            """
            
            # Define questions to answer
            questions = [
                "What is the expected annual return of this portfolio?",
                "Is this portfolio appropriate for my risk tolerance?",
                "How does my investment horizon affect this portfolio?",
                "What are the main risks in this portfolio?",
                "Should I consider rebalancing this portfolio?"
            ]
            
            # Generate answers
            for question in questions:
                inputs = qa_tokenizer(question, context, return_tensors="pt")
                outputs = qa_model(**inputs)
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits) + 1
                answer = qa_tokenizer.convert_tokens_to_string(
                    qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
                )
                
                # Clean up the answer
                answer = answer.strip()
                
                # If answer is empty or too short, provide a fallback
                if len(answer) < 10:
                    if "expected annual return" in question.lower():
                        answer = f"{expected_return:.2f}%"
                    elif "risk tolerance" in question.lower():
                        answer = f"Yes, this portfolio is designed for {risk_tolerance.lower()} risk tolerance"
                    elif "investment horizon" in question.lower():
                        answer = f"Your {investment_term.lower()} horizon allows for appropriate risk exposure"
                    elif "risks" in question.lower():
                        answer = f"Market volatility, inflation, and {'cryptocurrency volatility' if include_crypto else 'interest rate changes'}"
                    elif "rebalancing" in question.lower():
                        answer = "Rebalance annually or when allocations drift more than 5% from targets"
                
                qa_insights[question] = answer
    
    except Exception as e:
        st.error(f"Error in Q&A generation: {str(e)}")
        # Provide simulated QA if there's an error
        qa_insights = {
            "What is the expected annual return of this portfolio?": f"{portfolio_metrics['expected_annual_return']*100:.2f}%",
            "Is this portfolio appropriate for my risk tolerance?": f"Yes, this portfolio is aligned with your {risk_tolerance.lower()} risk tolerance.",
            "How does my investment horizon affect this portfolio?": f"Your {investment_term.lower()} horizon allows for appropriate risk exposure.",
            "What are the main risks in this portfolio?": f"Market volatility, inflation, and {'cryptocurrency volatility' if include_crypto else 'interest rate changes'}.",
            "Should I consider rebalancing this portfolio?": "Rebalance annually or when allocations drift more than 5% from targets."
        }
    
    return qa_insights

def display_portfolio_summary(portfolio_metrics, categorized_portfolio, investment_amount, 
                             risk_tolerance, investment_term, use_sentiment_analysis=False, 
                             use_qa_analysis=False, sentiment_results=None, qa_insights=None):
    """
    Display a summary of the optimized portfolio with optional AI analysis.
    
    Args:
        portfolio_metrics (dict): Portfolio performance metrics
        categorized_portfolio (dict): Portfolio allocation by category
        investment_amount (float): Total investment amount
        risk_tolerance (str): User's risk tolerance level
        investment_term (str): User's investment term preference
        use_sentiment_analysis (bool): Whether to display sentiment analysis
        use_qa_analysis (bool): Whether to display Q&A insights
        sentiment_results (dict): Results from sentiment analysis
        qa_insights (dict): Results from Q&A analysis
    """
    st.header("Your Optimized Investment Portfolio")
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Expected Annual Return",
            f"{portfolio_metrics['expected_annual_return']*100:.2f}%",
            help="Estimated return based on historical performance"
        )
    
    with col2:
        st.metric(
            "Portfolio Volatility",
            f"{portfolio_metrics['expected_volatility']*100:.2f}%",
            help="A measure of portfolio risk"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{portfolio_metrics['sharpe_ratio']:.2f}",
            help="Risk-adjusted return (higher is better)"
        )
    
    # Calculate category totals
    category_totals = {}
    for category, assets in categorized_portfolio.items():
        category_total = sum(asset['dollar_amount'] for asset in assets.values())
        category_percentage = (category_total / investment_amount) * 100
        category_totals[category] = {
            'total': category_total,
            'percentage': category_percentage
        }
    
    # Display portfolio allocation chart
    st.subheader("Portfolio Allocation")
    
    # Prepare data for pie chart
    labels = list(category_totals.keys())
    sizes = [data['percentage'] for data in category_totals.values()]
    
    # Filter out categories with zero allocation
    non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
    filtered_labels = [labels[i] for i in non_zero_indices]
    filtered_sizes = [sizes[i] for i in non_zero_indices]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=filtered_labels,
        values=filtered_sizes,
        hole=.4,
        marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    )])
    
    fig.update_layout(
        title_text="Asset Class Allocation",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed allocation tables
    st.subheader("Detailed Asset Allocation")
    
    # Function to display category table
    def display_category_table(category, assets):
        if assets:
            st.write(f"**{category}** (${category_totals[category]['total']:.2f}, {category_totals[category]['percentage']:.1f}%)")
            
            # Create dataframe for display
            data = []
            for ticker, asset in assets.items():
                row_data = {
                    "Symbol": ticker,
                    "Name": asset['name'],
                    "Allocation ($)": f"${asset['dollar_amount']:.2f}",
                    "Allocation (%)": f"{asset['percentage']:.1f}%",
                    "Expected Return": f"{asset['expected_return']*100:.2f}%"
                }
                
                # Add sentiment if available
                
                if use_sentiment_analysis and sentiment_results and ticker in sentiment_results:
                    sentiment = sentiment_results[ticker]
                    row_data["Sentiment"] = sentiment['label'].title()
                
                data.append(row_data)
            
            df = pd.DataFrame(data)
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    # Display each category
    for category in ["Stocks", "ETFs", "Bonds", "Alternatives", "Crypto"]:
        if category in categorized_portfolio and categorized_portfolio[category]:
            display_category_table(category, categorized_portfolio[category])
    
    # Display AI-powered insights if enabled
    if use_sentiment_analysis and sentiment_results:
        st.subheader("🤖 AI Market Sentiment Analysis")
        
        # Count sentiment categories
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        for ticker, sentiment in sentiment_results.items():
            sentiment_counts[sentiment['label']] += 1
        
        # Display sentiment overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive Sentiment", sentiment_counts["positive"])
        with col2:
            st.metric("Neutral Sentiment", sentiment_counts["neutral"])
        with col3:
            st.metric("Negative Sentiment", sentiment_counts["negative"])
        
        # Display sentiment details for top assets
        st.write("### Sentiment Details for Key Assets")
        
        # Get top assets by allocation
        all_assets = []
        for category, assets in categorized_portfolio.items():
            for ticker, details in assets.items():
                all_assets.append((ticker, details['dollar_amount']))
        
        top_assets = sorted(all_assets, key=lambda x: x[1], reverse=True)[:5]
        
        for ticker, _ in top_assets:
            if ticker in sentiment_results:
                sentiment = sentiment_results[ticker]
                
                # Create color based on sentiment
                color = "#4CAF50" if sentiment['label'] == "positive" else "#FFC107" if sentiment['label'] == "neutral" else "#F44336"
                
                with st.expander(f"{ticker}: {sentiment['label'].title()} ({sentiment['score']:.1%})"):
                    st.markdown(f"""
                    **News/Context:** {sentiment['text']}
                    
                    **Sentiment Score:** <span style='color:{color};font-weight:bold;'>{sentiment['score']:.1%}</span>
                    
                    **Potential Impact:** {'This positive sentiment may support the asset price in the near term.' if sentiment['label'] == 'positive' else 'This neutral sentiment suggests stable performance in the near term.' if sentiment['label'] == 'neutral' else 'This negative sentiment could create near-term volatility or pressure on the asset price.'}
                    """, unsafe_allow_html=True)
    
    # Display AI Q&A insights if enabled
    if use_qa_analysis and qa_insights:
        st.subheader("🤖 AI Portfolio Insights")
        
        # Create columns for Q&A display
        col1, col2 = st.columns(2)
        
        # Display questions and answers in columns
        questions = list(qa_insights.keys())
        for i, question in enumerate(questions):
            # Alternate between columns
            with col1 if i % 2 == 0 else col2:
                with st.expander(question):
                    st.write(qa_insights[question])
    
    # Investment insights
    st.subheader("Investment Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Profile Summary:**
        - Investment Amount: ${investment_amount:,.2f}
        - Risk Tolerance: {risk_tolerance}
        - Investment Term: {investment_term}
        
        This portfolio is designed to align with your {risk_tolerance.lower()} risk tolerance
        and {investment_term.lower().split('(')[0].strip()} investment horizon.
        """)
    
    with col2:
        if risk_tolerance in ["Low", "Medium-Low"]:
            st.markdown("""
            **Conservative Strategy Notes:**
            - Focus on capital preservation with steady growth
            - Higher allocation to bonds for stability
            - Limited exposure to volatile assets
            """)
        elif risk_tolerance == "Medium":
            st.markdown("""
            **Balanced Strategy Notes:**
            - Equal focus on growth and stability
            - Diversified across multiple asset classes
            - Moderate risk with potential for moderate returns
            """)
        else:  # Medium-High or High
            st.markdown("""
            **Growth Strategy Notes:**
            - Focus on long-term capital appreciation
            - Higher allocation to stocks for growth potential
            - Higher volatility with potential for higher returns
            """)
    
    # Educational disclaimer
    st.info("""
    **Disclaimer:** This portfolio is generated based on historical data and general investment principles.
    Past performance does not guarantee future results. Consider consulting with a financial advisor
    before making investment decisions. The allocations shown are for educational purposes only.
    """)

def run_investgenie_app():
    """Main function to run the InvestGenie application."""
    # Configure the page
    configure_page()
    
    # Get user inputs including AI feature preferences
    inputs = get_user_inputs()
    
    if all(inputs):
        investment_amount, risk_tolerance, investment_term, include_crypto, use_sentiment_analysis, use_qa_analysis = inputs
        
        # Define asset universe
        asset_universe = get_asset_universe()
        
        # Calculate risk profile based on inputs
        risk_profile = calculate_risk_profile(risk_tolerance, investment_term, include_crypto)
        
        # Get detailed allocation
        detailed_allocation = get_detailed_allocation(risk_profile, asset_universe)
        
        # Collect all tickers for data fetching
        all_tickers = list(detailed_allocation.keys())
        
        # Fetch financial data
        financial_data = fetch_financial_data(all_tickers)
        
        # Calculate portfolio metrics
        portfolio_metrics, portfolio_allocation = calculate_portfolio_metrics(
            detailed_allocation, financial_data, investment_amount
        )
        
        # Categorize assets
        categorized_portfolio = categorize_assets(portfolio_allocation, asset_universe)
        
        # Run GenAI analysis if requested
        sentiment_results = None
        qa_insights = None
        
        if use_sentiment_analysis:
            sentiment_results = analyze_portfolio_sentiment(portfolio_allocation, financial_data)
        
        if use_qa_analysis:
            qa_insights = generate_portfolio_qa(portfolio_metrics, risk_tolerance, investment_term, include_crypto)
        
        # Display portfolio summary with AI insights
        display_portfolio_summary(
            portfolio_metrics, categorized_portfolio, investment_amount, 
            risk_tolerance, investment_term, use_sentiment_analysis,
            use_qa_analysis, sentiment_results, qa_insights
        )

# Entry point when running the script directly
if __name__ == "__main__":
    run_investgenie_app()
