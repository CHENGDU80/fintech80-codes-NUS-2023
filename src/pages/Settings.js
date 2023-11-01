import React, { useState } from 'react';
import { Box, Button, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { Link } from 'react-router-dom';

const ConfigurationsPage = () => {
  const [selectedBuyOrSellSide, setSelectedBuyOrSellSide] = useState('');
  const [selectedSectors, setSelectedSectors] = useState([]);
  const [selectedTypeOfFunds, setTypeOfFunds] = useState('');

  const buyOrSellSide = ['Buy side', 'Sell side'];
  const sectors = [
    'Energy',
    'Materials',
    'Industrials',
    'Utilities',
    'Healthcare',
    'Financials',
    'Consumer Discretionary',
    'Consumer Staples',
    'Information Technology',
    'Communication Services',
    'Real Estate'
  ];
  const typeOfFunds = [
    'Mutual Fund',
    'Exchange-Traded Fund (ETF)',
    'Index Fund',
    'Hedge Fund',
    'Money Market Fund',
    'Bond Fund',
    'Real Estate Investment Trust (REIT)',
    'Commodity Fund',
    'Sector Fund',
    'Target-Date Fund',
    'Soverign Wealth Fund'
  ];

  return (
    <>
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div>
          <div>
            <FormControl sx={{ m: 2, minWidth: 300 }}>
              <InputLabel id="single-select-label">Buy / Sell Side</InputLabel>
              <Select
                labelId="single-select-label"
                id="single-select"
                value={selectedBuyOrSellSide}
                label="Single Select"
                onChange={(e) => setSelectedBuyOrSellSide(e.target.value)}
              >
                {buyOrSellSide.map((option, index) => (
                  <MenuItem key={index} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </div>

          <div>
            <FormControl sx={{ m: 2, minWidth: 300 }}>
              <InputLabel id="multi-select-label">Sectors</InputLabel>
              <Select
                labelId="multi-select-label"
                id="multi-select"
                multiple
                value={selectedSectors}
                label="Multi Select"
                onChange={(e) => setSelectedSectors(e.target.value)}
              >
                {sectors.map((option, index) => (
                  <MenuItem key={index} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </div>

          <div>
            <FormControl sx={{ m: 2, minWidth: 300 }}>
              <InputLabel id="single-select-label">Type of Investment Fund</InputLabel>
              <Select
                labelId="single-select-label"
                id="single-select"
                value={selectedTypeOfFunds}
                label="Single Select"
                onChange={(e) => setTypeOfFunds(e.target.value)}
              >
                {typeOfFunds.map((option, index) => (
                  <MenuItem key={index} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </div>

          <div>
            <Link to="/dashboard">
              <Button>
                <span>Save</span>
              </Button>
            </Link>
          </div>
        </div>
      </Box>
    </>
  );
};

export default ConfigurationsPage;
