// project import
import Routes from 'routes';
import ThemeCustomization from 'themes';
import ScrollTop from 'components/ScrollTop';
import NUSightGPT from 'components/NUSightGPT';

// ==============================|| APP - THEME, ROUTER, LOCAL  ||============================== //

const App = () => (
  <ThemeCustomization>
    <ScrollTop>
      <Routes />
      <NUSightGPT />
    </ScrollTop>
  </ThemeCustomization>
);

export default App;
