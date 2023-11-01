import { lazy } from 'react';

// project import
import Loadable from 'components/Loadable';
import MainLayout from 'layout/MainLayout';
import Configurations from 'pages/Settings';
import Companies from 'pages/Companies';
import Dashboard from 'pages/Dashboard';
import Action from 'pages/Action';

// render - dashboard
const Portfolio = Loadable(lazy(() => import('pages/portfolio')));

// ==============================|| MAIN ROUTING ||============================== //

const MainRoutes = {
  path: '/',
  element: <MainLayout />,
  children: [
    {
      path: '/',
      element: <Portfolio />
    },
    {
      path: 'portfolio',
      element: <Portfolio />
    },
    {
      path: 'dashboard',
      element: <Dashboard />
    },
    {
      path: 'companies',
      element: <Companies />
    },
    {
      path: 'action',
      element: <Action />
    },
    {
      path: 'config',
      element: <Configurations />
    }
  ]
};

export default MainRoutes;
