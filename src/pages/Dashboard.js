import { useState, useEffect } from 'react';
import MainCard from 'components/MainCard';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Chip,
  Grid,
  Stack,
  Typography
  // FormControl,
  // InputLabel,
  // Select,
  // MenuItem
} from '@mui/material';
import { withStyles } from '@mui/styles';
import { DataGrid } from '@mui/x-data-grid';
import { CaretDownOutlined } from '@ant-design/icons';
import PropTypes from 'prop-types';
import Typewriter from 'typewriter-effect';
import Papa from 'papaparse';
// import dfIndustryConsumerSentiment from 'data/df_industry_consumer_sent.csv';
// import dfIndustryPharmaSentiment from 'data/df_industry_pharmacy_sent.csv';
import dfIndustryTechSentiment from 'data/df_industry_tech_sent.csv';
import dfMacroSentiment from 'data/df_macro_sent.csv';

// const consumerSummary =
//   'The consumer industry is facing a variety of challenges as the holiday season approaches. Price, free shipping, value, and promotions/deals are the leading factors for purchase decisions, and total UK retail sales have increased by 4.1%. Retailers are focusing on seasonal staff, flexible shifts, online integration, and supply chain visibility to better manage costs and carbon emissions. Consumers are looking for virtual experiences, and consumer confidence is eroding as affordability pressures increase. Direct-to-Retail, social commerce, and supply chain management are also important trends in the consumer industry.';
// const pharmaceuticalsSummary =
//   "The pharmacy industry is rapidly evolving, with new treatments, therapies, and technologies being developed to address a wide range of medical conditions. Recent developments include GSK's Jemperli PD-1/L1 endometrial cancer immunotherapy, a competing risk joint model for dealing with different types of missing data in an intervention trial in prodromal Alzheimer's disease, Boan Biotech's Nivolumab Injection (BA1104) in China, Novartis' experimental drug atrasentan for a rare type of kidney disease, Ga(NO3)3 for the treatment of chronic lung infections, and the global Clinical Trial Supply and Logistics market. Additionally, there are a number of new therapies and pipelines in development, such as Pfizer-BioNTech's COVID-19 vaccine, Amplia Therapeutics' Phase 1B ACCENT trial in pancreatic cancer, Burning Rock's brigimadlin, Laverock Therapeutics' gene silencing platform, Georgiamune and Verily's strategic partnership to advance novel therapeutics for cancer, Alterity Therapeutics' ATH434, and DelveInsight's Sickle Cell Disease Pipeline Insight 2023. These developments are helping to improve diagnosis, treatment, and research.";

const technologySummary =
  "The technology industry is seeing a surge in activity, with China's tech behemoths Xiaomi and Huawei intensifying their efforts to integrate their diverse products, India proposing a roadmap to facilitate the development and transfer of environmentally sound technology, and Apple hosting its second product event of the season. Reliance Jio has announced that it will not increase consumer tariffs, even with the introduction of 5G technology. Standard Chartered's crypto security firm, Zodia Custody, is launching its services in Hong Kong. Brinc and Hatcher+ have announced their intention to identify, invest in, and support the growth of 300 new climate tech startups driving innovation over the next three years. The Federal Reserve is expected to stand pat at its policy meeting this week, with futures implying a 97% chance of rates staying at 5.25-5.5%. Nasdaq's forecast for 2024 for the capital markets here and abroad has remained optimistic.";

const macroSummaryShort =
  'Uncertainty about the economic outlook and ambiguous data are causing a slowdown in late 2022 and early 2023. Business cycles refer to the rise and fall of economic growth, and monetary policy plays a crucial role in influencing the performance of various financial assets. Inflation trends are expected to gradually decrease, and central banks are under pressure to further tighten their policies to rein in inflation.';

const macroSummaryLong =
  'The macroeconomic environment is currently in a state of uncertainty due to ambiguous data and a slowdown in late 2022 and early 2023. The US and China are the two locomotives of the global economy, and their accelerating growth could be a sign that the expansion is set to resume in 2024. Business cycles refer to the rise and fall of economic growth that an economy experiences over a period of time, and they are important for economists to predict economic events. Economic growth is distinguished from economic development, with the former referring to economies already experiencing rising per capita incomes.The global economic upswing that began around mid-2016 has become broader and stronger, and inflation is expected to gradually decrease. Monetary policy plays a crucial role in influencing the performance of various financial assets, and it is used to reflate the worldâ€™s third-largest economy. Interest rates have caused a sharp drop in UK household wealth, and the Federal Reserve is expected to make a decision this week on interest rates. Unemployment refers to the state of being without a job, and it is a measure of the percentage of the labor force that is unemployed. Skimpflation is a term used to describe a situation where companies respond to inflation by reducing the quantity of the products they offer.';

const StyledAccordion = withStyles({
  root: {
    border: '1px solid #e0e0e0',
    boxShadow: 'none',
    '&:not(:last-child)': {
      borderBottom: 'none'
    },
    '&:before': {
      display: 'none'
    }
  }
})(Accordion);

const StyledAccordionSummary = withStyles({
  root: {
    backgroundColor: '#f5f5f5',
    borderBottom: '1px solid #e0e0e0',
    marginBottom: -1,
    minHeight: 56,
    '&$expanded': {
      minHeight: 56
    }
  },
  content: {
    '&$expanded': {
      margin: '12px 0'
    }
  },
  expanded: {}
})(AccordionSummary);

const StyledAccordionDetails = withStyles((theme) => ({
  root: {
    padding: theme.spacing(2)
  }
}))(AccordionDetails);

const TypewriterWrapper = ({ text }) => (
  <Typewriter
    onInit={(typewriter) => {
      typewriter.changeDelay(0.01).typeString(text).start();
    }}
  />
);

const NewsSummaryAccordion = ({ title, sentimentScore, csvFile, initExpanded = false, hasMenuSelect = false }) => {
  const [summaryLength, setSummaryLength] = useState(50);
  const [expanded, setExpanded] = useState(initExpanded);
  const toggleExpanded = () => setExpanded(!expanded);

  // const lengthOfSummaryOptions = [50, 200];

  return (
    <StyledAccordion expanded={expanded}>
      <StyledAccordionSummary expandIcon={<CaretDownOutlined />} onClick={toggleExpanded}>
        <div style={{ width: '50%' }}>
          <Typography variant="h4" color="inherit">
            {title}
          </Typography>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', width: '50%', justifyContent: 'flex-end', marginRight: '1.2rem' }}>
          <Typography>Sentiment score:</Typography>
          <Chip
            variant="combined"
            color={sentimentScore <= 0.5 ? 'error' : 'success'}
            label={sentimentScore}
            sx={{ ml: 1.25 }}
            size="small"
          />
        </div>
      </StyledAccordionSummary>
      <StyledAccordionDetails>
        {/* {hasMenuSelect && (
          <FormControl sx={{ m: 2, minWidth: 300 }}>
            <InputLabel id="single-select-label">Length of Summary</InputLabel>
            <Select
              labelId="single-select-label"
              id="single-select"
              value={summaryLength}
              label="Single Select"
              onChange={(e) => setSummaryLength(e.target.value)}
            >
              {lengthOfSummaryOptions.map((option, index) => (
                <MenuItem key={index} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )} */}

        <Box sx={{ pt: 2.25 }}>
          {title == 'Technology' ? <TypewriterWrapper text={technologySummary} /> : <TypewriterWrapper text={macroSummaryShort} />}
        </Box>
        <CSVDataGrid csvFile={csvFile} />
      </StyledAccordionDetails>
    </StyledAccordion>
  );
};

NewsSummaryAccordion.propTypes = {
  title: PropTypes.string.isRequired,
  sentimentScore: PropTypes.number.isRequired,
  csvFile: PropTypes.string.isRequired,
  initExpanded: PropTypes.bool.isRequired
};

const CSVDataGrid = ({ csvFile }) => {
  const [rows, setRows] = useState([]);
  const [columns, setColumns] = useState([]);

  const BoldCell = (params) => <b>{params.value}</b>;

  useEffect(() => {
    const fetchData = async () => {
      Papa.parse(csvFile, {
        header: true,
        dynamicTyping: true,
        download: true,
        complete: (parsedData) => {
          const columns = [
            {
              field: 'Title',
              headerName: 'Title',
              width: 300,
              renderCell: BoldCell
            },
            {
              field: 'URL',
              headerName: 'URL',
              width: 200
            },
            {
              field: 'Description',
              headerName: 'Description',
              width: 250
            },
            {
              field: 'sentiment',
              headerName: 'Sentiment',
              width: 160,
              renderCell: (params) => (
                <Chip variant="combined" color={params.value === 'positive' ? 'info' : 'warning'} label={params.value} size="small" />
              )
            },
            {
              field: 'confidence',
              headerName: 'Confidence',
              width: 160,
              valueGetter: (params) => params.row.confidence.toFixed(2)
            }
          ];
          setColumns(columns);
          setRows(parsedData.data);
        },
        error: (error) => {
          console.error('CSV parsing error: ', error);
        }
      });
    };

    fetchData();
  }, [csvFile]);

  return (
    <div style={{ width: '100%' }}>
      <DataGrid
        rows={rows}
        columns={columns}
        initialState={{
          pagination: {
            paginationModel: {
              pageSize: 3
            }
          }
        }}
        filterModel={{
          items: [{ field: 'sentiment', operator: 'equals', value: 'positive' }]
        }}
        sortModel={[
          {
            field: 'confidence',
            sort: 'desc'
          }
        ]}
        pageSizeOptions={[3, 5, 10]}
        disableSelectionOnClick
        sx={{
          '&.MuiDataGrid-root .MuiDataGrid-cell:focus-within, &.MuiDataGrid-root .MuiDataGrid-columnHeader:focus-within': {
            outline: 'none !important'
          }
        }}
      />

      <DataGrid
        rows={rows}
        columns={columns}
        initialState={{
          pagination: {
            paginationModel: {
              pageSize: 3
            }
          }
        }}
        filterModel={{
          items: [{ field: 'sentiment', operator: 'equals', value: 'negative' }]
        }}
        sortModel={[
          {
            field: 'confidence',
            sort: 'desc'
          }
        ]}
        pageSizeOptions={[3, 5, 10]}
        disableSelectionOnClick
        sx={{
          '&.MuiDataGrid-root .MuiDataGrid-cell:focus-within, &.MuiDataGrid-root .MuiDataGrid-columnHeader:focus-within': {
            outline: 'none !important'
          }
        }}
      />
    </div>
  );
};

CSVDataGrid.propTypes = {
  csvFile: PropTypes.string.isRequired
};

const Companies = () => {
  return (
    <Grid container spacing={5} flexDirection="column">
      <Grid item>
        <MainCard contentSX={{ p: 2.25 }}>
          <Stack spacing={0.5}>
            <Typography variant="h6" color="textSecondary">
              {'Macro'}
            </Typography>
            <NewsSummaryAccordion title="Macro" sentimentScore={0.25} csvFile={dfMacroSentiment} initExpanded hasMenuSelect />
          </Stack>
        </MainCard>
      </Grid>

      <Grid item>
        <MainCard contentSX={{ p: 2.25 }}>
          <Stack spacing={0.5}>
            <Typography variant="h6" color="textSecondary">
              {'Industry'}
            </Typography>
            {/* 
          <NewsSummaryAccordion title="Consumer" text={consumerSummary} sentimentScore={0.43} csvFile={dfIndustryConsumerSentiment} />
          <NewsSummaryAccordion
            title="Pharmaceuticals"
            text={pharmaceuticalsSummary}
            sentimentScore={0.88}
            csvFile={dfIndustryPharmaSentiment} 
            />*/}
            <NewsSummaryAccordion title="Technology" sentimentScore={0.71} csvFile={dfIndustryTechSentiment} initExpanded />
          </Stack>
        </MainCard>
      </Grid>
    </Grid>
  );
};

export default Companies;
