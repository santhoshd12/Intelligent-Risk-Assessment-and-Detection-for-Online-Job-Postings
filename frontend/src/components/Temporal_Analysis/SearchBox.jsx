import React from 'react';

const SearchBox = ({ searchQuery, setSearchQuery, onSearch }) => {
  return (
    <div className="ta-card">
      <div className="ta-search">
        <input
          type="text"
          placeholder="Enter job title or description..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && onSearch()}
        />
        <button onClick={onSearch}>Search</button>
      </div>
    </div>
  );
};

export default SearchBox;
