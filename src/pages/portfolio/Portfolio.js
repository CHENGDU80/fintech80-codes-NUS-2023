import { Grid, Chip } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import { RiseOutlined, FallOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';

const PortfolioPage = () => {
  // const BoldHeader = (params) => <b>{params.colDef.headerName}</b>;
  const BoldLinkedCell = (params) => (
    <b>
      <Link to="/dashboard" style={{ textDecoration: 'none' }}>
        {params.value}
      </Link>
    </b>
  );

  // Sample company data for the portfolio
  const companies = [
    { name: 'Apple Inc', country: 'US', sector: 'Technology', market: 'NASDAQ', priceChange: -0.7 },
    { name: 'BYD Co Ltd', country: 'CN', sector: 'Manufacturing', market: 'HKEX', priceChange: 1.2 },
    { name: 'Wilmar International', country: 'SG', sector: 'Agriculture', market: 'SGX', priceChange: 0.5 },
    { name: 'Nvidia Corp', country: 'US', sector: 'Technology', market: 'NASDAQ', priceChange: 2.3 },
    { name: 'Tencent Holdings Ltd', country: 'CN', sector: 'Technology', market: 'HKEX', priceChange: -0.4 },
    { name: 'Samsung Electronics', country: 'KR', sector: 'Technology', market: 'KRX', priceChange: -0.6 },
    { name: 'Toyota Motor Corporation', country: 'JP', sector: 'Manufacturing', market: 'TSE', priceChange: 1.9 },
    { name: 'Unilever PLC', country: 'UK', sector: 'Consumer Goods', market: 'LSE', priceChange: -0.2 },
    { name: 'Microsoft Corporation', country: 'US', sector: 'Technology', market: 'NASDAQ', priceChange: 2.8 },
    { name: 'Alibaba Group Holding Ltd', country: 'CN', sector: 'Technology', market: 'NYSE', priceChange: 3.9 }
  ];

  // Column definitions for the DataGrid
  const columns = [
    { field: 'name', headerName: 'Company Name', width: 250, renderCell: BoldLinkedCell },
    { field: 'country', headerName: 'Country', width: 200 },
    { field: 'sector', headerName: 'Sector', width: 270 },
    { field: 'market', headerName: 'Market', width: 250 },
    {
      field: 'priceChange',
      headerName: 'Price Change',
      width: 180,
      renderCell: (params) => (
        <Grid item>
          <Chip
            variant="combined"
            color={params.row.priceChange <= 0 ? 'error' : 'success'}
            icon={
              <>
                {params.row.priceChange > 0 && <RiseOutlined style={{ fontSize: '0.75rem', color: 'inherit' }} />}
                {params.row.priceChange <= 0 && <FallOutlined style={{ fontSize: '0.75rem', color: 'inherit' }} />}
              </>
            }
            label={`${params.row.priceChange}%`}
            sx={{ ml: 1.25, pl: 1 }}
            size="small"
          />
        </Grid>
      )
    }
  ];

  return (
    <div style={{ height: 400, width: '100%' }}>
      <DataGrid
        rows={companies}
        columns={columns}
        pageSize={5}
        rowsPerPageOptions={[5, 10, 20]}
        getRowId={(row) => row.name + row.country}
        sx={{
          '&.MuiDataGrid-root .MuiDataGrid-cell:focus-within, &.MuiDataGrid-root .MuiDataGrid-columnHeader:focus-within': {
            outline: 'none !important'
          }
        }}
      />
    </div>
  );
};

export default PortfolioPage;
